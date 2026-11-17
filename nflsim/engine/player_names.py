from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from nflsim.names import display_name


def player_names(*tables: pd.DataFrame) -> dict[str, str]:
    names = {}
    for table in tables:
        if "player_id" not in table:
            continue
        # Later candidates win: nickname + last name ("Pat Mahomes") over legal first + last.
        candidates = [table[column] for column in ("player_name", "full_name", "display_name") if column in table]
        if {"first_name", "last_name"} <= set(table):
            candidates.append(table.first_name.astype("string") + " " + table.last_name.astype("string"))
        if "last_name" in table:
            candidates.append(pd.Series([display_name(row.get("football_name"), row.get("last_name"), row.get("first_name"))
                                         if pd.notna(row.get("last_name")) else None
                                         for row in table.to_dict("records")], index=table.index, dtype="object"))
        for candidate in candidates:
            for player_id, name in zip(table.player_id, candidate):
                if pd.isna(player_id) or pd.isna(name):
                    continue
                identity, label = str(player_id), str(name).strip()
                if label and label != identity and label.lower() not in {"nan", "none", "<na>"}:
                    names[identity] = label
    return names


class NamedPlayerTables(HTMLParser):
    def __init__(self, names):
        super().__init__(convert_charrefs=False)
        self.names = names
        self.parts = []
        self.in_cell = False
        self.replacements = 0

    def handle_starttag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())
        if tag == "td":
            self.in_cell = True

    def handle_startendtag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")
        if tag == "td":
            self.in_cell = False

    def handle_data(self, data):
        key = data.strip()
        if self.in_cell and key in self.names:
            self.parts.append(data.replace(key, html.escape(self.names[key])))
            self.replacements += 1
        else:
            self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_comment(self, data):
        self.parts.append(f"<!--{data}-->")

    def handle_decl(self, decl):
        self.parts.append(f"<!{decl}>")


def name_report_tables(report: str, names: dict[str, str]) -> tuple[str, int]:
    parser = NamedPlayerTables(names)
    parser.feed(report)
    parser.close()
    return "".join(parser.parts), parser.replacements


def main():
    parser = argparse.ArgumentParser(description="Refresh player display names in saved HTML tables without changing simulations")
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    roster = pd.read_parquet(args.roster) if args.roster.suffix == ".parquet" else pd.read_csv(args.roster)
    names = player_names(roster)
    for path in args.reports:
        report, replacements = name_report_tables(path.read_text(encoding="utf-8"), names)
        path.write_text(report, encoding="utf-8")
        print(f"{path}: {replacements} table labels resolved")


if __name__ == "__main__":
    main()