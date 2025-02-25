import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import anthropic
import fitz
import requests
from dotenv import load_dotenv
import aiohttp


class PDFParser:
    def __init__(self, model="claude-3-5-sonnet-latest"):
        """Initialise les paramètres et API Keys"""
        load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

        self.model = model
        self.pages_per_chunk = 5
        self.max_size = 10 * 1024 * 1024  # 10MB
        self.max_workers = 8
        self.retry_delay = 10
        self.max_retries = 5
        self.temp_pdf_dir = "temp_pdfs"

        # Création du dossier temporaire
        os.makedirs(self.temp_pdf_dir, exist_ok=True)

    def download_pdf(self, url):
        """Télécharge un PDF et le stocke temporairement"""
        filename = os.path.join(self.temp_pdf_dir, os.path.basename(url))
        try:
            response = requests.get(url, stream=True, verify=False)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return filename
            else:
                print(
                    f"⚠️ Impossible de télécharger {url} (Code: {response.status_code})"
                )
        except requests.RequestException as e:
            print(f"❌ Erreur téléchargement {url}: {e}")
        return None

    async def extract_text_from_pdf(self, pdf_path):
        """Extrait et fusionne le texte d'un PDF via Claude 3.5 en chunks"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        tasks = []
        for i in range(0, total_pages, self.pages_per_chunk):
            end_page = min(i + self.pages_per_chunk, total_pages)
            tasks.append(self._process_pdf_chunk(pdf_path, i, end_page))

        extracted_texts = []
        results = await asyncio.gather(*tasks)
        for result in results:
            if result:
                extracted_texts.append(result["text"])

        os.remove(pdf_path)  # Suppression après traitement
        return "\n\n".join(extracted_texts)  # Fusion du texte

    async def _process_pdf_chunk(self, pdf_path, start_page, end_page):
        """Découpe un PDF en chunks et envoie à Claude 3.5"""
        doc = fitz.open(pdf_path)
        sub_doc = fitz.open()

        for j in range(start_page, end_page):
            sub_doc.insert_pdf(doc, from_page=j, to_page=j)

        pdf_bytes = sub_doc.write()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        if len(pdf_bytes) > self.max_size:
            print(f"⚠️ Chunk {start_page + 1}-{end_page} trop grand, ignoré.")
            return None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": pdf_base64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Extract only the raw text from this PDF section in French, with no additional comments."
                                }
                            ]
                        }
                    ],
                )

                if isinstance(response.content, list):
                    extracted_text = " ".join(
                        item.text if hasattr(item, "text") else str(item)
                        for item in response.content
                    )
                else:
                    extracted_text = (
                        response.content.text
                        if hasattr(response.content, "text")
                        else response.content
                    )

                return {
                    "start_page": start_page + 1,
                    "end_page": end_page,
                    "text": extracted_text,
                }

            except anthropic.APIStatusError as e:
                if "429" in str(e):
                    wait_time = self.retry_delay * attempt
                    print(
                        f"⏳ Rate limit dépassé (tentative {attempt}). Pause de {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print(f"❌ Erreur pages {start_page + 1}-{end_page}: {str(e)}")
                    return None

        return None

    async def process_pdf(self, pdf_url):
        """Pipeline complet pour traiter un seul PDF et retourner le texte fusionné"""
        pdf_path = self.download_pdf(pdf_url)
        if pdf_path:
            return await self.extract_text_from_pdf(pdf_path)
        else:
            return None
