"""Rebuildable SQLite index for project media metadata."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Iterable, Mapping, Sequence


INDEX_FILENAME = ".yolo26_media.sqlite3"


class MediaIndex:
    """Keep a small, disposable index beside a project's original annotations."""

    def __init__(self, project_path: str | Path) -> None:
        self.path = Path(project_path) / INDEX_FILENAME

    def _connect(self) -> sqlite3.Connection:
        try:
            return self._open_connection()
        except sqlite3.DatabaseError:
            # The index is cache only. Rebuild it instead of risking project data.
            for candidate in (self.path, self.path.with_name(self.path.name + "-wal"),
                              self.path.with_name(self.path.name + "-shm")):
                candidate.unlink(missing_ok=True)
            return self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS media (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    annotation_count INTEGER NOT NULL DEFAULT 0,
                    thumbnail_key TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            return connection
        except Exception:
            connection.close()
            raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back an operation, then release the SQLite file handle."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def sync_snapshot(
        self, paths: Iterable[str], annotation_counts: Mapping[str, int], source: str = "project"
    ) -> None:
        unique_paths = list(dict.fromkeys(paths))
        rows: list[tuple[object, ...]] = []
        for media_path in unique_paths:
            try:
                stat = os.stat(media_path)
            except OSError:
                continue
            thumbnail_key = f"{stat.st_size:x}-{stat.st_mtime_ns:x}"
            rows.append(
                (
                    media_path,
                    stat.st_size,
                    stat.st_mtime_ns,
                    "image",
                    None,
                    None,
                    int(annotation_counts.get(media_path, 0)),
                    thumbnail_key,
                    source,
                )
            )
        with self._connection() as connection:
            connection.execute("DELETE FROM media")
            connection.executemany(
                """
                INSERT INTO media(path, size, mtime_ns, media_type, width, height, annotation_count,
                                  thumbnail_key, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def upsert(self, paths: Sequence[str], annotation_counts: Mapping[str, int]) -> None:
        if not paths:
            return
        with self._connection() as connection:
            for media_path in paths:
                try:
                    stat = os.stat(media_path)
                except OSError:
                    continue
                connection.execute(
                    """
                    INSERT INTO media(path, size, mtime_ns, media_type, width, height, annotation_count,
                                      thumbnail_key, source)
                    VALUES (?, ?, ?, 'image', NULL, NULL, ?, ?, 'imported')
                    ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns,
                        annotation_count=excluded.annotation_count, thumbnail_key=excluded.thumbnail_key
                    """,
                    (media_path, stat.st_size, stat.st_mtime_ns, int(annotation_counts.get(media_path, 0)),
                     f"{stat.st_size:x}-{stat.st_mtime_ns:x}"),
                )

    def clear(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM media")

    def count(self) -> int:
        try:
            with self._connection() as connection:
                return int(connection.execute("SELECT COUNT(*) FROM media").fetchone()[0])
        except sqlite3.Error:
            return 0
