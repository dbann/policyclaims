#!/usr/bin/env python3

#this script fetches abstracts from scopus for a list of journals and saves them in json files
#it uses the Scopus API to fetch the data
#it saves the data in json files in the json_files folder

#import necessary libraries
import argparse
import requests
import time
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY")
INST_TOKEN = os.getenv("INST_TOKEN", "")
print("SCOPUS_API_KEY assigned:", SCOPUS_API_KEY)

SCOPUS_API_URL = "https://api.elsevier.com/content/search/scopus"
JSON_DIR = Path("json_files")
JSON_DIR.mkdir(exist_ok=True)

# Lower this if you risk hitting rate limits
PAUSE_SECONDS = 0.2  

# Map Scopus "ID" or custom ID to ISSN + an abbreviation
JOURNAL_ISSNS = {
    # Economics
 #   "S145200904": {"issn": "1057-9230", "abbrev": "health_econ"}, 
 #   "S16530434": {"issn": "1618-7598", "abbrev": "eur_j_health_econ"},    
 #   "S166621295": {"issn": "0167-6296", "abbrev": "j_health_econ"},   
 #   "S23340": {"issn": "1570-677X", "abbrev": "econ_human_bio"},       
 #   "S21100891323": {"issn": "2332-3493", "abbrev": "am_j_health_econ"},

    # Epidemiology and Public Health
    "S27024": {"issn": "0300-5771", "abbrev": "int_j_epidemiol"},       
    "S20224": {"issn": "0749-3797", "abbrev": "am_j_prev_med"},       
    "S20040": {"issn": "0091-7435", "abbrev": "prev_med"},       
    "S15470582": {"issn": "1044-3983", "abbrev": "epidemiology"},    
    "S156988948": {"issn": "0143-005X", "abbrev": "j_epidemiol_comm_health"},   
    "S2764808104": {"issn": "2468-2667", "abbrev": "lancet_public_health"},  
    "S168049282": {"issn": "0090-0036", "abbrev": "am_j_public_health"},   
    "S4210220588": {"issn": "1101-1262", "abbrev": "eur_j_public_health"},  
    "S170967050": {"issn": "0002-9262", "abbrev": "am_j_epidemiol"},   
    "S48690275": {"issn": "0393-2990", "abbrev": "eur_j_epidemiol"},

    # Health Policy
  #  "S15926": {"issn": "0278-2715", "abbrev": "health_affairs"},       
  #  "S22415": {"issn": "0887-378X", "abbrev": "milbank_q"},       
  #  "S22871": {"issn": "0197-5897", "abbrev": "j_public_health_policy"},       
  #  "S21306": {"issn": "0168-8510", "abbrev": "health_policy"},

    # General Medical Journals
  #  "S22515": {"issn": "0028-4793", "abbrev": "nejm"},       
  #  "S21060": {"issn": "0959-8138", "abbrev": "bmj"},       
  #  "S22528": {"issn": "0140-6736", "abbrev": "lancet"},       
  #  "S15840": {"issn": "0098-7484", "abbrev": "jama"},       
  #  "S17929": {"issn": "1549-1676", "abbrev": "plos_med"},

    # Other social science
  #  "S21334": {"issn": "0070-3370", "abbrev": "demography"},       
  #  "S123166236": {"issn": "0141-9889", "abbrev": "sociol_health_ill"},   
  #  "S106822843": {"issn": "0277-9536", "abbrev": "soc_sci_med"},   
  #  "S2764672113": {"issn": "2352-8273", "abbrev": "ssm_pop_health"},
}

def fetch_total_results(issn, year):
    """
    Make a quick query with small count to get the total number of results.
    We do this by specifying the year in the query: PUBYEAR = year
    """
    query = f"ISSN({issn}) AND PUBYEAR = {year} AND DOCTYPE(ar)"
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY, "Accept": "application/json"}
    params = {
        "query": query,
        "count": 1,           # just need minimal response
        "view": "STANDARD",   # standard view is enough to get total
    }
    try:
        resp = requests.get(SCOPUS_API_URL, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        total_str = data["search-results"]["opensearch:totalResults"]
        return int(total_str)
    except Exception as e:
        print(f"[fetch_total_results] Error for ISSN={issn}, year={year}: {e}")
        return 0


def fetch_scopus_works_for_year(issn, year, cursor="*"):
    """
    Fetch all results for ISSN in a single year. Return both:
    - list of processed records
    - the final cursor (if you want to do something with it).
    """
    all_year_records = []
    while True:
        query = f"ISSN({issn}) AND PUBYEAR = {year} AND DOCTYPE(ar)"
        headers = {"X-ELS-APIKey": SCOPUS_API_KEY, "Accept": "application/json"}
        params = {
            "query": query,
            "cursor": cursor,
            "view": "COMPLETE"
        }
        try:
            resp = requests.get(SCOPUS_API_URL, headers=headers, params=params)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[fetch_scopus_works_for_year] Scopus fetch failed (ISSN={issn}, year={year}): {e}")
            break
        
        data = resp.json()
        search_results = data.get("search-results", {})
        entries = search_results.get("entry", [])
        if not entries:
            break  # No more data
        
        # Process each entry
        for entry in entries:
            processed = process_scopus_work(entry)
            if processed:
                all_year_records.append(processed)
        
        # Next cursor
        next_cursor = search_results.get("cursor", {}).get("@next")
        if not next_cursor or len(entries) < 25:
            # No further pages or we got fewer than 25, meaning last page
            break
        else:
            cursor = next_cursor

        # Pause to respect rate limits
        time.sleep(PAUSE_SECONDS)

    return all_year_records, cursor


def process_scopus_work(entry):
    """Extract metadata from Scopus entry, including abstract, etc."""
    try:
        doi = entry.get("prism:doi", "")
        title = entry.get("dc:title", "")
        pub_year_full = entry.get("prism:coverDate", "Unknown")  # e.g. 2021-05-10
        pub_year = pub_year_full.split("-")[0] if pub_year_full else "Unknown"
        journal = entry.get("prism:publicationName", "")
        abstract = entry.get("dc:description", "")
        affiliations = entry.get('affiliation', [])
        country = "Unknown"
        if affiliations:
            country = affiliations[0].get('affiliation-country', 'Unknown')

        # Keywords
        keywords = entry.get('authkeywords', '')
        if keywords:
            keywords = keywords.split(' | ')
        else:
            keywords = []

        cited_by = entry.get('citedby-count', 0)
        scopus_id = entry.get("dc:identifier", "Unknown")

        return {
            "scopus_id": scopus_id,
            "doi": doi,
            "title": title,
            "journal": journal,
            "publication_year": pub_year,
            "keywords": keywords,
            "abstract": abstract,
            "article_type": entry.get("subtypeDescription", "").lower(),
            "corresponding_author_country": country,
            "cited_by_count": cited_by
        }
    except Exception as e:
        print(f"[process_scopus_work] Error: {e}")
        return None


def load_existing_journal_data(journal_id, abbrev):
    """
    Load the JSON if it exists. Return a list of existing items
    plus a set of (scopus_id, or doi) for quick deduplication.
    """
    filename = JSON_DIR / f"{abbrev}_{journal_id}.json"
    if not filename.exists():
        return [], set()
    with open(filename, "r", encoding="utf-8") as f:
        records = json.load(f)
    # Build a set of something unique (DOI or scopus_id)
    existing_ids = set()
    for r in records:
        # prefer scopus_id if it’s there
        sid = r.get("scopus_id")
        if sid:
            existing_ids.add(sid)
        else:
            existing_ids.add(r.get("doi", ""))
    return records, existing_ids


def save_journal_data(journal_id, abbrev, records):
    """
    Saves final records to JSON for that journal.
    """
    filename = JSON_DIR / f"{abbrev}_{journal_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    print(f"[save_journal_data] Saved {len(records)} articles to {filename}")


def fetch_or_load_journal(journal_id, journal_info, refresh=False):
    """
    Loop over years. For each year, see if we already have enough results in
    the existing JSON. If not, fetch more. Append to the store, deduplicating
    as needed.
    """
    issn = journal_info["issn"]
    abbrev = journal_info["abbrev"]

    # Load existing data if not refresh
    if not refresh:
        all_records, existing_ids = load_existing_journal_data(journal_id, abbrev)
    else:
        all_records, existing_ids = [], set()

    # Go year by year to avoid hitting the 5,000 limit
    for year in range(1990, 2025):
        # Check how many we already have for this year
        existing_for_year = [r for r in all_records if r.get("publication_year") == str(year)]
        count_for_year = len(existing_for_year)

        total_for_year = fetch_total_results(issn, year)
        if total_for_year <= count_for_year:
            print(f"[{abbrev} {year}] Already have {count_for_year}/{total_for_year}. Skipping.")
            continue
        
        print(f"[{abbrev} {year}] Found {count_for_year} existing. Need {total_for_year} total. Fetching...")
        # Fetch in pages via cursor
        # We always start a fresh cursor because there's no reliable way
        # to skip to the 'middle' of pagination with Scopus. We'll simply
        # deduplicate if we get duplicates.
        year_records, _ = fetch_scopus_works_for_year(issn, year, cursor="*")

        # Add to the main store (deduplicate by scopus_id or doi).
        new_count = 0
        for rec in year_records:
            sid = rec.get("scopus_id", "")
            if sid not in existing_ids:
                existing_ids.add(sid)
                all_records.append(rec)
                new_count += 1

        print(f"[{abbrev} {year}] +{new_count} new articles fetched. Total = {len(all_records)}")

        # Save incremental updates to reduce data loss if something breaks
        save_journal_data(journal_id, abbrev, all_records)

    return all_records


def save_processed_data(refresh=False):
    for journal_id, journal_info in JOURNAL_ISSNS.items():
        print("=" * 50)
        print(f"Processing: {journal_id} ({journal_info['abbrev']})")
        print("=" * 50)
        fetch_or_load_journal(journal_id, journal_info, refresh=refresh)
    print("All done.")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Scopus journal articles, year by year.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached files and start fresh.")
    args = parser.parse_args()

    save_processed_data(refresh=args.refresh)