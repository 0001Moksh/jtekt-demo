
import os
import re
from pathlib import Path

import msal
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GRAPH = "https://graph.microsoft.com/v1.0"
SITE_URL = os.environ["SHAREPOINT_SITE_URL"].rstrip("/")
FOLDER_NAME = os.getenv("SHAREPOINT_FOLDER_NAME", "JTEKT")
CACHE_DIR = BASE_DIR / ".cache" / "sharepoint"

SUPPORTED = {".pdf", ".pptx"}


def get_token() -> str:
    app = msal.ConfidentialClientApplication(
        client_id=os.environ["MS_CLIENT_ID"],
        authority=(
            "https://login.microsoftonline.com/"
            + os.environ["MS_TENANT_ID"]
        ),
        client_credential=os.environ["MS_CLIENT_SECRET"],
    )
    result = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )
    if "access_token" not in result:
        raise RuntimeError(
            "Microsoft Graph authentication failed: "
            + result.get("error_description", "Unknown error")
        )
    return result["access_token"]


class Graph:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}"
        })

    def get_json(self, url: str, params=None):
        response = self.session.get(
            url, params=params, timeout=60
        )
        if not response.ok:
            raise RuntimeError(
                f"Microsoft Graph returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        return response.json()

    def get_all(self, url: str):
        while url:
            data = self.get_json(url)
            yield from data.get("value", [])
            url = data.get("@odata.nextLink")

    def find_jtekt_folder(self, drive_id: str):
        """Search the document library recursively for JTEKT."""
        root = f"{GRAPH}/drives/{drive_id}/root/children"
        stack = list(self.get_all(root))

        while stack:
            item = stack.pop()
            if "folder" not in item:
                continue

            if item.get("name", "").casefold() == FOLDER_NAME.casefold():
                return item

            children_url = (
                f"{GRAPH}/drives/{drive_id}/items/"
                f"{item['id']}/children"
            )
            stack.extend(self.get_all(children_url))

        return None


def download_jtekt_documents() -> list[dict]:
    token = get_token()
    graph = Graph(token)

    # Resolve the SharePoint site from its hostname and server-relative path.
    from urllib.parse import urlparse

    parsed = urlparse(SITE_URL)
    site_path = parsed.path.rstrip("/")
    site = graph.get_json(
        f"{GRAPH}/sites/{parsed.netloc}:{site_path}"
    )
    site_id = site["id"]

    drives = list(graph.get_all(
        f"{GRAPH}/sites/{site_id}/drives"
    ))
    if not drives:
        raise RuntimeError("No document libraries were found on the site.")

    matches = []
    for drive in drives:
        folder = graph.find_jtekt_folder(drive["id"])
        if folder:
            matches.append((drive, folder))

    if not matches:
        raise RuntimeError(
            f"Could not find a folder named '{FOLDER_NAME}' "
            "in the site's document libraries."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple '{FOLDER_NAME}' folders in different "
            "libraries. Configure the exact document library/folder path."
        )

    drive, folder = matches[0]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    def visit(folder_item):
        children_url = (
            f"{GRAPH}/drives/{drive['id']}/items/"
            f"{folder_item['id']}/children"
        )

        for item in graph.get_all(children_url):
            if "folder" in item:
                visit(item)
                continue

            name = item.get("name", "")
            if Path(name).suffix.lower() not in SUPPORTED:
                continue

            # Include item ID to avoid filename collisions.
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            local_path = CACHE_DIR / f"{item['id']}_{safe_name}"

            content_url = (
                f"{GRAPH}/drives/{drive['id']}/items/"
                f"{item['id']}/content"
            )
            response = graph.session.get(
                content_url, timeout=120
            )
            if not response.ok:
                raise RuntimeError(
                    f"Download failed for {name}: "
                    f"HTTP {response.status_code}"
                )

            local_path.write_bytes(response.content)
            downloaded.append({
                "path": local_path,
                "name": name,
                "item_id": item["id"],
                "source_url": item.get("webUrl", ""),
            })

    visit(folder)

    if not downloaded:
        raise RuntimeError(
            f"No PDF or PPTX files found in '{FOLDER_NAME}'."
        )

    return downloaded
