"""Workspace-scoped topic bookmarks, navigation and append-only chat transcripts."""

import json
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from ..utils.summarizer import KeySentenceExtractor
from ..utils.text_processor import TextProcessor


class TopicConflict(ValueError):
    """The branch, navigation cursor or retry identity changed."""


_RESUME = re.compile(
    r"(?:回到|切回|转回|说回|聊回|接着(?:上次|写|讨论|做)|接回|之前那个|上次那个|继续(?:上次|之前|写|讨论|做))"
)
_PAST_REFERENCE = re.compile(
    r"^(?:(?:请问|还记得|我想问问|我想问|再说说|说回|回到|切回|接着|继续|关于)\s*)?(?:我们)?"
    r"(?:上次|那次|先前|之前)(?:我们)?"
    r"(?:聊(?:到|起)?|谈(?:论|到)?|提(?:起|到|及|过)|说(?:到|起|过)|讨论|记录|研究|做过|写过|"
    r"(?=那个|那件|那份|那条|的作业|作业))(?:过)?(?:的)?"
    r"(?:作业(?:[：:，,]\s*|(?:里|中)(?:提到|讨论|记录)?的?))?"
    r"(?:有关|关于)?(?:那个|那件|那条|那份|那段)?"
)
_BACK = re.compile(r"^(?:请)?(?:返回|回到|切回|退回)(?:上一个|上一|上个|刚才的)(?:话题|分支)[。？！?！\s]*$")
_NEW = re.compile(
    r"^(?:我们|今天|现在|接下来)?\s*(?:新建话题|开一个话题|开个话题|新话题|"
    r"(?:换个话题|换个话题来|换个话题，)?(?:来聊聊|聊一聊|聊聊|聊)|换到|换个话题)"
    r"\s*[：:]?\s*[《「“]?([^。，,；;？！?！\n》」”]{1,80})"
)
_NOTE = re.compile(r"(?:不(?:要)?(?:切换|切走|换)(?:当前)?话题|继续当前话题|话题先不换|只记在.{1,50}里)")
_PROGRESS = re.compile(r"(?:当前进度|停在|当前位置)[：:]\s*([^。；;\n]{1,200})")
_HERE = re.compile(r"(?:回到|继续|接着)(?:我|我们)?(?:手头|眼下|当前|眼前)的?(?:事|话题|工作|任务)")
_SELECT = re.compile(
    r"^(?:我们|今天|现在)?(?:回到|切回|转回|聊聊|说回|换到|再看|我指的是|我说的是|就是|选|是)"
)


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 256:
        raise ValueError(f"{label} must be a nonempty string of at most 256 characters")
    return value.strip()


def _mentions(text, names):
    """Keep only whole labels; a longer label shadows the substring inside it."""
    folded = text.casefold()
    hits = []
    for name in names:
        pattern = re.escape(name.casefold())
        if name[0].isascii() and name[0].isalnum():
            pattern = r"(?<![a-z0-9_])" + pattern
        if name[-1].isascii() and name[-1].isalnum():
            pattern += r"(?![a-z0-9_])"
        hits.extend((m.start(), m.end(), name) for m in re.finditer(pattern, folded))
    return list(
        dict.fromkeys(
            name for a, b, name in hits if not any(c <= a and b <= d and d - c > b - a for c, d, _ in hits)
        )
    )


def _position(text):
    match = _PROGRESS.search(text)
    return match[1].strip() if match else None


def _evidence_anchor(text, reference):
    """Use a literal subject or action. Shared fragments and question boilerplate cannot vote."""
    if re.search(r"不是|并非|不要|或者|还是|也许|可能是", text):
        return ""
    tail = (
        text[reference.end() :]
        if reference
        else re.sub(r"^(?:我(?:说|指)的是|我记得|就是|选|是)?(?:那个|那段|那次|那件)?", "", text)
    )
    clue = re.split(
        r"[。；;？！?！,，\n]|后来|接下来|然后|接着怎么|最终|现在怎样|现在怎么|怎样|怎么样|"
        r"怎么了|怎么样了|如何|是什么|是多少|做了什么|发生了什么",
        tail,
        maxsplit=1,
    )[0].strip(" ：:《》“”「」\"' ")
    clue = re.sub(r"(?:的?那(?:一)?(?:段|次|条|件|个)(?:事|动作)?|的动作|的事情|呢)$", "", clue)
    distinct = re.sub(
        r"当时|刚才|那个|这个|之前|上次|那次|先前|前面|以前|一样|上面|下面|同一|"
        r"有关|相关|内容|事情|东西|问题|项目|任务|工作|作业|话题|动作|原来|那段|进度|"
        r"登记码|编号|框数|数值|数字|多少|什么|怎么|一下",
        "",
        clue,
    )
    if not re.fullmatch(r"[一-鿿a-zA-Z0-9._+ -]{4,64}", clue):
        return ""
    if len(re.findall(r"[一-鿿]", distinct)) >= 4 or re.search(
        r"(?=[\w.+-]*\d)[A-Za-z][A-Za-z0-9.+_-]{3,}", distinct
    ):
        return clue
    return ""


class TopicWorkspaceStore:
    def __init__(self, storage_root):
        folder = Path(storage_root)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "topic_workspaces.sqlite3"
        self.extractor = KeySentenceExtractor()
        with closing(self._connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS topic_workspaces (
                    workspace TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0,
                    trail TEXT NOT NULL DEFAULT '[]', pending TEXT);
                CREATE TABLE IF NOT EXISTS topic_provider_cache (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, signature TEXT NOT NULL,
                    sent_at REAL NOT NULL, prefix_blocks INTEGER NOT NULL, PRIMARY KEY(workspace,topic));
                CREATE TABLE IF NOT EXISTS topic_prefix_runs (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, signature TEXT NOT NULL,
                    sent_at REAL NOT NULL, missed_tokens INTEGER NOT NULL, tail_floor INTEGER NOT NULL, PRIMARY KEY(workspace,topic));
                CREATE TABLE IF NOT EXISTS topic_reading_focus (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, revision INTEGER NOT NULL,
                    source_revision INTEGER NOT NULL, scene TEXT NOT NULL, edition TEXT NOT NULL,
                    PRIMARY KEY(workspace,topic));
                CREATE TABLE IF NOT EXISTS topic_clarifications (
                    workspace TEXT, request_id TEXT, user TEXT, fingerprint TEXT, reply TEXT,
                    PRIMARY KEY(workspace,request_id));
                CREATE TABLE IF NOT EXISTS topic_branches (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                    position TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(workspace,topic));
                CREATE TABLE IF NOT EXISTS topic_events (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, revision INTEGER NOT NULL,
                    kind TEXT NOT NULL, user TEXT NOT NULL, assistant TEXT,
                    wire_content TEXT, source TEXT, request_id TEXT, fingerprint TEXT,
                    PRIMARY KEY(workspace,topic,revision), UNIQUE(workspace,request_id));
                CREATE INDEX IF NOT EXISTS topic_recent_model_turns ON topic_events(workspace) WHERE kind='turn';
                CREATE TABLE IF NOT EXISTS topic_message_refs (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, message_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL, role TEXT NOT NULL,
                    PRIMARY KEY(workspace,topic,message_id), UNIQUE(workspace,topic,revision,role));
                CREATE TABLE IF NOT EXISTS topic_quotes (
                    id INTEGER PRIMARY KEY, workspace TEXT NOT NULL, topic TEXT NOT NULL,
                    message_id INTEGER NOT NULL, quote TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS topic_quote_messages ON topic_quotes(workspace,topic,message_id);
                CREATE INDEX IF NOT EXISTS topic_source_revisions ON topic_events(workspace,topic,revision)
                    WHERE kind='source';
                CREATE TABLE IF NOT EXISTS topic_source_coverage (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, revision INTEGER NOT NULL,
                    PRIMARY KEY(workspace, topic));
                CREATE TABLE IF NOT EXISTS topic_terms (
                    workspace TEXT NOT NULL, topic TEXT NOT NULL, term TEXT NOT NULL, quote_id INTEGER NOT NULL,
                    PRIMARY KEY(workspace,topic,term,quote_id)) WITHOUT ROWID;
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def _session(self, write=False):
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db

    @staticmethod
    def _state(db, workspace):
        row = db.execute("SELECT * FROM topic_workspaces WHERE workspace=?", (workspace,)).fetchone()
        if row is None:
            return 0, [], None
        return (
            row["revision"],
            json.loads(row["trail"]),
            json.loads(row["pending"]) if row["pending"] else None,
        )

    @staticmethod
    def _save_state(db, workspace, trail, pending=None):
        db.execute(
            """INSERT INTO topic_workspaces(workspace,revision,trail,pending) VALUES(?,1,?,?)
            ON CONFLICT(workspace) DO UPDATE SET revision=revision+1, trail=excluded.trail, pending=excluded.pending""",
            (
                workspace,
                json.dumps(trail, ensure_ascii=False),
                json.dumps(pending, ensure_ascii=False) if pending else None,
            ),
        )

    @staticmethod
    def _ensure_branch(db, workspace, topic):
        db.execute("INSERT OR IGNORE INTO topic_branches(workspace,topic) VALUES(?,?)", (workspace, topic))

    @staticmethod
    def _metadata(db, workspace, topic):
        row = db.execute(
            "SELECT topic,revision,position,source FROM topic_branches WHERE workspace=? AND topic=?",
            (workspace, topic),
        ).fetchone()
        return dict(row) if row else {"topic": topic, "revision": 0, "position": "", "source": ""}

    def _listing(self, db, workspace):
        revision, trail, pending = self._state(db, workspace)
        rows = db.execute(
            "SELECT topic,revision,position,source FROM topic_branches WHERE workspace=? ORDER BY topic",
            (workspace,),
        )
        breadcrumb = self._snapshot(db, workspace, trail[-1], False)["breadcrumb"] if trail else [workspace]
        return {
            "current": trail[-1] if trail else None,
            "breadcrumb": breadcrumb,
            "topics": [dict(row) for row in rows],
            "workspace_revision": revision,
            "pending": pending,
        }

    def list_topics(self, workspace_id):
        workspace = _identifier(workspace_id, "workspace_id")
        with self._session() as db:
            return self._listing(db, workspace)

    def open_topic(self, workspace_id, topic, *, position=None, source=None):
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        with self._session(write=True) as db:
            self._ensure_branch(db, workspace, topic)
            _, trail, _ = self._state(db, workspace)
            if not trail or trail[-1] != topic:
                trail.append(topic)
            if position is not None or source is not None:
                db.execute(
                    """UPDATE topic_branches SET position=coalesce(?,position), source=coalesce(?,source)
                    WHERE workspace=? AND topic=?""",
                    (position, source, workspace, topic),
                )
            self._save_state(db, workspace, trail)
            return self._listing(db, workspace)

    def back(self, workspace_id):
        workspace = _identifier(workspace_id, "workspace_id")
        with self._session(write=True) as db:
            _, trail, pending = self._state(db, workspace)
            if len(trail) > 1:
                trail.pop()
                self._save_state(db, workspace, trail)
            elif pending:
                self._save_state(db, workspace, trail)
            return self._listing(db, workspace)

    def bootstrap(self, workspace_id, topic, messages):
        """Import only the preceding turns of a previously unseen workspace."""
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        with self._session(write=True) as db:
            if (
                self._state(db, workspace)[0]
                or db.execute(
                    "SELECT 1 FROM topic_branches WHERE workspace=? LIMIT 1", (workspace,)
                ).fetchone()
            ):
                return False
            self._ensure_branch(db, workspace, topic)
            self._save_state(db, workspace, [topic])
            revision = 0
            for msg in messages:
                role = msg["role"]
                if role not in ("user", "assistant") or not isinstance(msg["content"], str):
                    raise ValueError("bootstrap accepts user/assistant text messages")
                revision += 1
                db.execute(
                    """INSERT INTO topic_events(workspace,topic,revision,kind,user,assistant)
                    VALUES (?,?,?,?,?,?)""",
                    (
                        workspace,
                        topic,
                        revision,
                        role,
                        msg["content"] if role == "user" else "",
                        msg["content"] if role == "assistant" else None,
                    ),
                )
            db.execute(
                "UPDATE topic_branches SET revision=? WHERE workspace=? AND topic=?",
                (revision, workspace, topic),
            )
            self._index_pending(db, workspace, topic)
            return True

    def _index_pending(self, db, workspace, topic):
        # A branch created by an earlier version has events but no message index.
        boundary = db.execute(
            """SELECT message_id,revision FROM topic_message_refs
                    WHERE workspace=? AND topic=? ORDER BY message_id DESC LIMIT 1""",
            (workspace, topic),
        ).fetchone()
        ordinal, revision = tuple(boundary) if boundary else (0, 0)
        rows = db.execute(
            """SELECT revision,kind,user,assistant,wire_content FROM topic_events
                        WHERE workspace=? AND topic=? AND revision>? ORDER BY revision""",
            (workspace, topic, revision),
        )
        for row in rows:
            parts = []
            if row["kind"] != "assistant":
                original = (
                    row["wire_content"] if row["kind"] == "note" and row["wire_content"] else row["user"]
                )
                parts.append(("user", original))
            if row["kind"] in ("turn", "assistant"):
                parts.append(("assistant", row["assistant"]))
            for role, content in parts:
                ordinal += 1
                db.execute(
                    "INSERT INTO topic_message_refs VALUES(?,?,?,?,?)",
                    (workspace, topic, ordinal, row["revision"], role),
                )
                seen = set()
                # Chunk contiguous input before extraction so even multiline quotes are
                # literal substrings of the original message.
                for offset in range(0, len(content), 512):
                    for quote in self.extractor.extract(content[offset : offset + 512]):
                        if quote in seen:
                            continue
                        seen.add(quote)
                        self._index_quote(db, workspace, topic, ordinal, quote)
        self._cover_sources(db, workspace, topic)

    @staticmethod
    def _index_quote(db, workspace, topic, message_id, quote):
        quote_id = db.execute(
            "INSERT INTO topic_quotes(workspace,topic,message_id,quote) VALUES(?,?,?,?)",
            (workspace, topic, message_id, quote),
        ).lastrowid
        db.executemany(
            "INSERT INTO topic_terms VALUES(?,?,?,?)",
            ((workspace, topic, term, quote_id) for term in TextProcessor.query_terms(quote)),
        )

    @staticmethod
    def _covered_source_revision(db, workspace, topic):
        row = db.execute(
            "SELECT revision FROM topic_source_coverage WHERE workspace=? AND topic=?", (workspace, topic)
        ).fetchone()
        return row[0] if row else 0

    def _cover_sources(self, db, workspace, topic):
        # Keep the usual key sentences, then cover the rest of each attached source.
        # An old archive is backfilled once, in the same transaction as its watermark.
        covered = self._covered_source_revision(db, workspace, topic)
        rows = db.execute(
            """SELECT e.revision,e.user,r.message_id FROM topic_events e
            JOIN topic_message_refs r ON r.workspace=e.workspace AND r.topic=e.topic
              AND r.revision=e.revision AND r.role='user'
            WHERE e.workspace=? AND e.topic=? AND e.kind='source' AND e.revision>? ORDER BY e.revision""",
            (workspace, topic, covered),
        )
        last = covered
        for last, content, ordinal in rows:
            seen = {
                r[0]
                for r in db.execute(
                    "SELECT quote FROM topic_quotes WHERE workspace=? AND topic=? AND message_id=?",
                    (workspace, topic, ordinal),
                )
            }
            for sentence in TextProcessor.split_sentences(content):
                # Keep exact substrings, including identifiers crossing a normal
                # window boundary inside an unpunctuated passage.
                for offset in range(0, len(sentence), 192):
                    quote = sentence[offset : offset + 256]
                    if quote not in seen:
                        seen.add(quote)
                        self._index_quote(db, workspace, topic, ordinal, quote)
                    if offset + 256 >= len(sentence):
                        break
        if last > covered:
            db.execute(
                """INSERT INTO topic_source_coverage VALUES(?,?,?)
                ON CONFLICT(workspace,topic) DO UPDATE SET revision=excluded.revision""",
                (workspace, topic, last),
            )

    @contextmanager
    def _indexed_session(self, workspace, topic):
        # On warm reads only touch the two small indexed boundaries. Avoid taking
        # a writer lock or loading unrelated source text to perform a lookup.
        with self._session() as db:
            revision = self._metadata(db, workspace, topic)["revision"]
            row = db.execute(
                """SELECT revision FROM topic_message_refs WHERE workspace=? AND topic=?
                              ORDER BY message_id DESC LIMIT 1""",
                (workspace, topic),
            ).fetchone()
            latest_source = (
                db.execute(
                    """SELECT max(revision) FROM topic_events
                WHERE workspace=? AND topic=? AND kind='source'""",
                    (workspace, topic),
                ).fetchone()[0]
                or 0
            )
            if (row[0] if row else 0) == revision and self._covered_source_revision(
                db, workspace, topic
            ) == latest_source:
                yield db
                return
        with self._session(write=True) as db:
            self._index_pending(db, workspace, topic)
            yield db

    def search(self, workspace_id, topic, query, limit=5):
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        if not isinstance(query, str) or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("expected query text and a limit between 1 and 50")
        terms = sorted(TextProcessor.query_terms(query))[:128]
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        with self._indexed_session(workspace, topic) as db:
            rows = db.execute(
                f"""WITH matches AS (
                SELECT quote_id,count(*) AS hits FROM topic_terms WHERE workspace=? AND topic=?
                AND term IN ({placeholders}) GROUP BY quote_id), ranked AS (
                SELECT q.message_id,q.quote,m.hits,row_number() OVER(
                    PARTITION BY q.message_id ORDER BY m.hits DESC,q.id ASC) AS choice
                FROM matches m JOIN topic_quotes q ON q.id=m.quote_id)
                SELECT k.message_id,k.quote,k.hits,r.role,r.revision,e.kind,e.source
                FROM ranked k JOIN topic_message_refs r ON r.workspace=? AND r.topic=? AND r.message_id=k.message_id
                JOIN topic_events e ON e.workspace=r.workspace AND e.topic=r.topic AND e.revision=r.revision
                WHERE choice=1 ORDER BY hits DESC,k.message_id DESC LIMIT ?""",
                (workspace, topic, *terms, workspace, topic, limit),
            )
            return [dict(row) for row in rows]

    def source_message(self, workspace_id, topic, message_id):
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        if type(message_id) is not int or message_id < 1:
            raise ValueError("message_id must be a positive integer")
        with self._indexed_session(workspace, topic) as db:
            row = db.execute(
                """SELECT r.message_id,r.role,r.revision,e.kind,e.source,
                CASE WHEN r.role='assistant' THEN e.assistant WHEN e.kind='note' THEN coalesce(e.wire_content,e.user)
                ELSE e.user END AS content FROM topic_message_refs r JOIN topic_events e
                ON e.workspace=r.workspace AND e.topic=r.topic AND e.revision=r.revision
                WHERE r.workspace=? AND r.topic=? AND r.message_id=?""",
                (workspace, topic, message_id),
            ).fetchone()
            if row is None:
                raise KeyError("message not found in topic")
            return dict(row)

    def snapshot(self, workspace_id, topic, *, include_events=False):
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        with self._session() as db:
            return self._snapshot(db, workspace, topic, include_events=include_events)

    def _snapshot(self, db, workspace, topic, include_messages=True, *, include_events=False):
        meta = self._metadata(db, workspace, topic)
        meta["breadcrumb"] = [workspace, topic] + ([meta["position"]] if meta["position"] else [])
        if not include_messages:
            return meta
        last_request = db.execute(
            """SELECT c.signature,c.sent_at,c.prefix_blocks,coalesce(d.missed_tokens,0) AS missed_tokens,
                          coalesce(d.tail_floor,0) AS tail_floor
                   FROM topic_provider_cache c LEFT JOIN topic_prefix_runs d
                   USING(workspace,topic,signature,sent_at) WHERE c.workspace=? AND c.topic=?""",
            (workspace, topic),
        ).fetchone()
        meta["provider_cache"] = dict(last_request) if last_request else None
        messages = []
        rows = db.execute(
            """SELECT kind,user,assistant,wire_content FROM topic_events WHERE workspace=? AND topic=?
                    ORDER BY revision""",
            (workspace, topic),
        )
        if include_events:
            meta["events"] = []
        for row in rows:
            if include_events:
                meta["events"].append({k: row[k] for k in ("kind", "user", "assistant")})
            if row["kind"] == "assistant":
                messages.append({"role": "assistant", "content": row["assistant"]})
                continue
            message = {"role": "user", "content": row["user"]}
            if row["wire_content"] is not None:
                message["wire_content"] = row["wire_content"]
            if row["kind"] == "note" and row["wire_content"]:
                message["content"] = row["wire_content"]
            messages.append(message)
            if row["kind"] == "turn":
                messages.append({"role": "assistant", "content": row["assistant"]})
        meta["messages"] = messages
        return meta

    def has_active_prefix_peer(self, workspace, topic, signature, since, now):
        """Check recent completed model turns before purchasing another cold opening."""
        with self._session() as db:
            return (
                db.execute(
                    """SELECT 1 FROM (
                       SELECT topic FROM topic_events WHERE workspace=? AND kind='turn'
                       ORDER BY rowid DESC LIMIT 4
                   ) recent JOIN topic_prefix_runs r ON r.workspace=? AND r.topic=recent.topic
                   JOIN topic_provider_cache c USING(workspace,topic,signature,sent_at)
                   WHERE r.topic<>? AND r.signature=? AND r.sent_at>? AND r.sent_at<=?
                   AND r.tail_floor>0 AND c.prefix_blocks>0 LIMIT 1""",
                    (workspace, workspace, topic, signature, since, now),
                ).fetchone()
                is not None
            )

    def _topics_with_anchor(self, db, workspace, candidates, anchor):
        terms = TextProcessor.query_terms(anchor)
        if not terms:
            return []
        # The postings narrow the search; the entire clue must occur in one quote.
        term = max(terms, key=lambda item: (len(item), item))
        pattern = re.escape(anchor)
        if anchor[0].isascii() and anchor[0].isalnum():
            pattern = r"(?<![a-z0-9_-])" + pattern
        if anchor[-1].isascii() and anchor[-1].isalnum():
            pattern += r"(?![a-z0-9_-])"
        matches = re.compile(pattern, re.I)
        selected = []
        for topic in candidates:
            self._index_pending(db, workspace, topic)
            rows = db.execute(
                """SELECT q.quote FROM topic_terms t JOIN topic_quotes q ON q.id=t.quote_id
                WHERE t.workspace=? AND t.topic=? AND t.term=? AND instr(lower(q.quote),lower(?))>0""",
                (workspace, topic, term, anchor),
            )
            if any(matches.search(row[0]) for row in rows):
                selected.append(topic)
        return selected

    def route(self, workspace_id, text, *, topic=None, request_id=None, fingerprint=None, allow_reading=True):
        workspace = _identifier(workspace_id, "workspace_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be nonempty")
        raw_text = text
        with self._session(write=True) as db:
            if request_id is not None:
                previous = self._replay(db, workspace, request_id, text, fingerprint)
                if previous and previous.get("status") == "clarify":
                    return previous["route"]
            version, trail, pending = self._state(db, workspace)
            if pending and not pending["candidates"]:
                self._save_state(db, workspace, trail)
                version += 1
                pending = None
            branches = self._listing(db, workspace)["topics"]
            names = [b["topic"] for b in branches]
            current = trail[-1] if trail else None
            text = text.strip()
            hits = _mentions(text, names)
            reference = _PAST_REFERENCE.match(text)
            aside = re.search(r"顺(?:口|便|带)(?:说|提|补|记)|旁线(?:补记|更正)|先记(?:下|一下)", text)
            correction = re.search(r"更正|改(?:为|成|到)|补记|更新|记(?:下|一下)|登记", text)
            explicit_open = bool(_NEW.match(text))
            other_named = any(name != current for name in hits)
            is_note = bool(
                not explicit_open
                and (_NOTE.search(text) or (current and other_named and aside and correction))
            )
            if is_note:
                # A trailing "继续代码" describes the existing focus, not a second
                # destination for the note.
                note_part = re.split(r"[，,。；;]\s*(?:我们)?(?:还是)?继续", text, maxsplit=1)[0]
                hits = _mentions(note_part, names)
            status = "ready"
            question = text
            candidates = []
            back_requested = False
            implicit = None
            if allow_reading and topic is None and not is_note and os.getenv("HR_IMPLICIT_FOLLOWUPS") == "1":
                from .implicit_followup import resolve

                implicit = resolve(db, workspace, text, names, current, pending)
            if implicit:
                chosen = implicit["chosen"]
                question = implicit.get("question", text)
                candidates = implicit.get("pending", {}).get("candidates", [])
            elif topic is not None:
                chosen = _identifier(topic, "topic")
                if pending and chosen in pending["candidates"] and _SELECT.match(text):
                    question = pending["question"] + "\n用户补充：" + text
                    is_note = pending.get("note", False)
            elif _BACK.fullmatch(text) and len(trail) > 1:
                chosen = trail[-2]
                back_requested = True
            elif is_note:
                chosen = hits[0] if len(hits) == 1 else None
                if not chosen:
                    candidates = hits or names
            elif intro := _NEW.match(text):
                chosen = intro[1].strip().strip("《》「」“”")
            elif pending and len(hits) == 1 and hits[0] in pending["candidates"]:
                chosen = hits[0]
                question = pending["question"] + "\n用户补充：" + text
                is_note = pending.get("note", False)
            elif _HERE.search(text) and current:
                chosen = current
            elif len(hits) == 1 and (
                _SELECT.match(text)
                or _RESUME.search(text)
                or (reference and text[reference.end() :].startswith(hits[0]))
                or text.rstrip("。！？?") == hits[0]
                or text.startswith(hits[0])
                or not current
            ):
                chosen = hits[0]
            elif len(hits) > 1 and (not current or reference or _RESUME.search(text) or _SELECT.match(text)):
                chosen = None
                candidates = hits
            elif reference or _RESUME.search(text) or pending:
                pool = pending["candidates"] if pending and pending["candidates"] else names
                clues = [
                    b["topic"]
                    for b in branches
                    if b["topic"] in pool
                    and any(
                        len(piece) >= 2 and piece.casefold() in text.casefold()
                        for piece in re.split(r"\s*(?:/|›|>|、|，|：|:|；)\s*", b["position"])
                        if piece
                    )
                ]
                candidates = clues or list(pool)
                anchored = []
                if (reference or pending) and len(clues) != 1:
                    anchor = _evidence_anchor(text, reference)
                    if anchor:
                        anchored = self._topics_with_anchor(db, workspace, candidates, anchor)
                    if anchored:
                        candidates = anchored
                confirmed = bool(clues or anchored or not (reference or pending))
                chosen = candidates[0] if confirmed and len(candidates) == 1 else None
                if pending and chosen:
                    question = pending["question"] + "\n用户补充：" + text
                    is_note = pending.get("note", False)
            else:
                chosen = current or "默认话题"

            if chosen is None:
                status = "clarify"
                updated = {
                    "question": pending["question"] if pending else text,
                    "candidates": candidates,
                    "note": pending.get("note", False) if pending else is_note,
                }
                if implicit:
                    updated = implicit["pending"]
                if not candidates:
                    updated = None
                if updated != pending:
                    self._save_state(db, workspace, trail, updated)
                    version += 1
            elif is_note:
                status = "note"
            reply = {
                "status": status,
                "topic": chosen,
                "question": question,
                "candidates": candidates,
                "workspace_revision": version,
                "current": current,
                "back_requested": back_requested,
            }
            if implicit and chosen:
                implicit["branch_revision"] = self._metadata(db, workspace, chosen)["revision"]
                reply["reading"] = implicit
            if status == "clarify":
                label = "、".join(candidates)
                reply["clarification"] = (
                    f"你是指哪个话题（{label}）？" if label else "这个工作区还没有话题。你想先聊什么？"
                )
                if implicit:
                    reply["clarification"] = implicit["clarification"]
                if request_id is not None:
                    db.execute(
                        "INSERT INTO topic_clarifications VALUES(?,?,?,?,?)",
                        (workspace, request_id, raw_text, fingerprint, json.dumps(reply, ensure_ascii=False)),
                    )
            return reply

    @staticmethod
    def _replay(db, workspace, request_id, user, fingerprint):
        if request_id is None:
            return None
        row = db.execute(
            """SELECT topic,revision,user,assistant,fingerprint FROM topic_events
                      WHERE workspace=? AND request_id=?""",
            (workspace, request_id),
        ).fetchone()
        if row is None:
            clarify = db.execute(
                "SELECT user,fingerprint,reply FROM topic_clarifications WHERE workspace=? AND request_id=?",
                (workspace, request_id),
            ).fetchone()
            if clarify is None:
                return None
            if clarify["user"] != user or clarify["fingerprint"] != fingerprint:
                raise TopicConflict("request_id already used with different input")
            result = json.loads(clarify["reply"])
            return {
                "status": "clarify",
                "topic": None,
                "assistant": result["clarification"],
                "revision": result["workspace_revision"],
                "candidates": result["candidates"],
                "route": result,
            }
        if row["user"] != user or row["fingerprint"] != fingerprint:
            raise TopicConflict("request_id already used with different input")
        return {"topic": row["topic"], "assistant": row["assistant"], "revision": row["revision"]}

    def replay(self, workspace_id, request_id, user, *, fingerprint=None):
        workspace = _identifier(workspace_id, "workspace_id")
        with self._session() as db:
            return self._replay(db, workspace, request_id, user, fingerprint)

    def append_turn(
        self,
        workspace_id,
        topic,
        user,
        assistant,
        *,
        expected_revision,
        activate=True,
        position=None,
        source=None,
        wire_content=None,
        expected_workspace_revision=None,
        request_id=None,
        fingerprint=None,
        include_messages=True,
        provider_cache=None,
        reading_focus=None,
    ):
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        return self._append(
            workspace,
            topic,
            "turn",
            user,
            assistant,
            expected_revision,
            activate,
            position if position is not None else _position(user),
            source,
            wire_content,
            expected_workspace_revision,
            request_id,
            fingerprint,
            include_messages,
            provider_cache,
            reading_focus,
        )

    def add_note(
        self,
        workspace_id,
        topic,
        text,
        *,
        source=None,
        expected_workspace_revision=None,
        request_id=None,
        fingerprint=None,
        include_messages=True,
        acknowledgement=None,
        wire_content=None,
    ):
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        return self._append(
            workspace,
            topic,
            "note",
            text,
            acknowledgement or f"已记入「{topic}」，当前话题未切换。",
            None,
            False,
            None,
            source,
            wire_content,
            expected_workspace_revision,
            request_id,
            fingerprint,
            include_messages,
        )

    def attach_source(self, workspace_id, topic, text, *, source, request_id=None):
        """Attach an immutable reference without selecting or advancing its topic."""
        workspace = _identifier(workspace_id, "workspace_id")
        topic = _identifier(topic, "topic")
        if not isinstance(text, str) or not text.strip() or not isinstance(source, str) or not source.strip():
            raise ValueError("source and text must be nonempty")
        return self._append(
            workspace,
            topic,
            "source",
            text,
            None,
            None,
            False,
            None,
            source,
            "引用资料（来源：" + source + "）\n" + text,
            None,
            request_id,
            "source:" + source,
            False,
        )

    def _append(
        self,
        workspace,
        topic,
        kind,
        user,
        assistant,
        expected_revision,
        activate,
        position,
        source,
        wire_content,
        expected_workspace_revision,
        request_id,
        fingerprint,
        include_messages,
        provider_cache=None,
        reading_focus=None,
    ):
        with self._session(write=True) as db:
            replay = self._replay(db, workspace, request_id, user, fingerprint)
            if replay:
                if replay["topic"] != topic:
                    raise TopicConflict("request_id already used for another topic")
                return self._snapshot(db, workspace, topic, include_messages)
            version, trail, pending = self._state(db, workspace)
            if expected_workspace_revision is not None and expected_workspace_revision != version:
                raise TopicConflict("workspace navigation changed; retry against the current workspace")
            self._ensure_branch(db, workspace, topic)
            old = self._metadata(db, workspace, topic)
            if expected_revision is not None and old["revision"] != expected_revision:
                raise TopicConflict("topic changed; retry against the current revision")
            rev = old["revision"] + 1
            db.execute(
                """INSERT INTO topic_events(workspace,topic,revision,kind,user,assistant,wire_content,source,request_id,fingerprint)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    workspace,
                    topic,
                    rev,
                    kind,
                    user,
                    assistant,
                    wire_content if wire_content != user else None,
                    source,
                    request_id,
                    fingerprint,
                ),
            )
            db.execute(
                """UPDATE topic_branches SET revision=?, position=coalesce(?,position),source=coalesce(?,source)
                       WHERE workspace=? AND topic=?""",
                (rev, position, None if kind == "source" else source, workspace, topic),
            )
            if reading_focus is not None:
                db.execute(
                    """INSERT INTO topic_reading_focus VALUES(?,?,?,?,?,?)
                       ON CONFLICT(workspace,topic) DO UPDATE SET revision=excluded.revision,
                       source_revision=excluded.source_revision,scene=excluded.scene,edition=excluded.edition""",
                    (
                        workspace,
                        topic,
                        rev,
                        reading_focus["source_revision"],
                        reading_focus["scene"],
                        reading_focus["edition"],
                    ),
                )
            if provider_cache is not None:
                db.execute(
                    """INSERT INTO topic_provider_cache VALUES (?,?,?,?,?)
                    ON CONFLICT(workspace,topic) DO UPDATE SET signature=excluded.signature,
                    sent_at=excluded.sent_at,prefix_blocks=excluded.prefix_blocks""",
                    (workspace, topic, *provider_cache[:3]),
                )
                if len(provider_cache) == 5:
                    signature, sent, _, missed, floor = provider_cache
                    db.execute(
                        """INSERT INTO topic_prefix_runs VALUES (?,?,?,?,?,?)
                           ON CONFLICT(workspace,topic) DO UPDATE SET signature=excluded.signature,
                           sent_at=excluded.sent_at,missed_tokens=excluded.missed_tokens,tail_floor=excluded.tail_floor""",
                        (workspace, topic, signature, sent, missed, floor),
                    )
            self._index_pending(db, workspace, topic)
            if activate:
                if len(trail) > 1 and trail[-2] == topic and _BACK.fullmatch(user.strip()):
                    trail.pop()
                elif not trail or trail[-1] != topic:
                    trail.append(topic)
                self._save_state(db, workspace, trail)
            elif (
                kind == "note"
                and pending
                and pending.get("note")
                and topic in pending["candidates"]
                and wire_content
                and pending["question"] in wire_content
            ):
                self._save_state(db, workspace, trail)
            return self._snapshot(db, workspace, topic, include_messages)
