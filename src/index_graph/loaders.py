from typing import List
from langchain_core.documents import Document
import requests
from datetime import datetime


class CodeProjectLoader:
    """Loader for research documents from RD-AGRI API."""

    def __init__(self, project_code: str):
        self.project_code = project_code
        self.base_url = "https://rd-agri.fr/rest/search/getResults"

    def load(self) -> List[Document]:
        """Load documents from the RD-AGRI API."""
        # Prepare the search payload
        payload = {
            "fullText": "",
            "motsCles": [],
            "enrichPath": [],
            "organismes": [],
            "typesDonnees": ["DOCUMENT"],
            "codeProjet": [],
            "annees": [],
            "page": 0,
            "searchSize": 2000,
            "tri": "DATE",
        }

        # Make the API request
        response = requests.post(self.base_url, json=payload, verify=False)
        data = response.json()

        print(data)
        # Convert results to Documents
        documents = []
        for result in data.get("results", []):
            # Only process DOCUMENT type and ensure URL exists and ends with .pdf
            url_document = result.get("urlDocument")
            if url_document and url_document.lower().endswith('.pdf'):
                # Create metadata
                metadata = {
                    "id": result.get("id"),
                    "title": result.get("titre"),
                    "publication_year": result.get("anneePublication"),
                    "publisher": result.get("publicateur"),
                    "url": result.get("urlDocument"),
                    "project_code": result.get("codeProjet"),
                }

                # Create Document object
                doc = Document(
                    page_content=result.get("description", ""), metadata=metadata
                )
                documents.append(doc)

        return documents
