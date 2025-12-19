"""
Research Engine with Qdrant for semantic search
"""
import os
import requests
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


class ResearchEngine:
    def __init__(self):
        """Initialize the research engine"""
        print("Initializing Research Engine...")
        
        # Validate Groq API Key
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not configured in .env")
        print("Groq API Key configured")
        
        # Qdrant Cloud client
        qdrant_host = os.getenv("QDRANT_HOST")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_host:
            raise ValueError("QDRANT_HOST not configured in .env")
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY not configured in .env")
        
        try:
            print(f"Connecting to Qdrant Cloud: {qdrant_host}")
            
            self.qdrant = QdrantClient(
                url=f"https://{qdrant_host}",
                api_key=qdrant_api_key,
                timeout=60,
                prefer_grpc=False
            )
            
            collections = self.qdrant.get_collections()
            print(f"Qdrant Cloud connected successfully")
            print(f"   Existing collections: {len(collections.collections)}")
            
        except Exception as e:
            print(f"Error connecting to Qdrant Cloud: {e}")
            raise
        
        # Embeddings model
        print("Loading embeddings model...")
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embeddings model loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self._setup_collection()
        print("Research Engine ready to use")
    
    def _setup_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        collection_name = "research_docs"
        
        try:
            self.qdrant.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists")
        except:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created")
    
    def fetch_arxiv_papers(self, query: str, max_results: int = 15) -> List[Dict]:
        """
        Search papers in arXiv API with improved query handling
        """
        print(f"Searching papers in arXiv: '{query}'")
        
        base_url = "http://export.arxiv.org/api/query"
        
        # Clean and prepare query - search in title and abstract
        clean_query = query.replace("latest", "").replace("solutions", "").strip()
        
        params = {
            'search_query': f'ti:{clean_query} OR abs:{clean_query}',  # Search in title OR abstract
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',  # Changed from 'submittedDate' to 'relevance'
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            
            papers = []
            entries = response.text.split('<entry>')
            
            for entry in entries[1:]:
                try:
                    title = entry.split('<title>')[1].split('</title>')[0].strip()
                    title = ' '.join(title.split())
                    
                    summary = entry.split('<summary>')[1].split('</summary>')[0].strip()
                    summary = ' '.join(summary.split())[:700]
                    
                    link = entry.split('<id>')[1].split('</id>')[0].strip()
                    
                    papers.append({
                        'title': title,
                        'content': summary,
                        'url': link,
                        'source': 'arXiv'
                    })
                except Exception as e:
                    continue
            
            print(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def store_in_qdrant(self, documents: List[Dict]):
        """
        Store documents in Qdrant with embeddings
        
        Args:
            documents: List of documents with title, content and URL
        """
        if not documents:
            print("No documents to store")
            return
        
        print(f"Storing {len(documents)} documents in Qdrant...")
        
        try:
            # Generate embeddings
            texts = [f"{doc['title']} {doc['content']}" for doc in documents]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            
            # Create points for Qdrant
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                points.append(PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        'title': doc['title'],
                        'content': doc['content'],
                        'url': doc['url'],
                        'source': doc.get('source', 'unknown')
                    }
                ))
            
            # Insert into Qdrant
            self.qdrant.upsert(
                collection_name="research_docs",
                points=points
            )
            
            print(f"{len(points)} documents stored successfully")
            
        except Exception as e:
            print(f"Error storing in Qdrant: {e}")
    
    def search_relevant_docs(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search relevant documents using semantic search
        
        Args:
            query: User search query
            top_k: Number of results to return
            
        Returns:
            List of most relevant documents
        """
        print(f"Searching relevant documents for: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode([query], show_progress_bar=False)[0]
            
            # Search in Qdrant
            results = self.qdrant.search(
                collection_name="research_docs",
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            # Format results
            relevant_docs = []
            for hit in results:
                relevant_docs.append({
                    'title': hit.payload['title'],
                    'content': hit.payload['content'],
                    'url': hit.payload['url'],
                    'source': hit.payload['source'],
                    'relevance_score': round(hit.score, 3)
                })
            
            print(f"Found {len(relevant_docs)} relevant documents")
            return relevant_docs
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def generate_report(self, query: str, documents: List[Dict]) -> str:
        """
        Generate research report using Groq API with requests
        
        Args:
            query: Research question
            documents: List of relevant documents
            
        Returns:
            Report in text format
        """
        print("Generating report with Groq (Llama 3.1) via requests...")
        
        # Prepare context with maximum 5 documents
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            context_parts.append(
                f"[{i}] {doc['title']}\n"
                f"{doc['content']}\n"
                f"Source: {doc['url']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Model prompt
        prompt = f"""You are an expert academic research assistant.

RESEARCH QUESTION:
{query}

ACADEMIC PAPERS FOUND:
{context}

INSTRUCTIONS:
1. Generate a professional executive report of maximum 400 words
2. Summarize the most important findings
3. Identify key trends and patterns
4. Use references [1], [2], [3], etc. to cite sources
5. Write in clear and professional English
6. DO NOT invent information not present in the sources

REPORT:"""
        
        try:
            # Call to Groq API with requests - CORRECTED FORMAT
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 1200,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            # Debug info if fails
            if response.status_code != 200:
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            report = data['choices'][0]['message']['content']
            
            print("Report generated successfully")
            return report
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {response.status_code}"
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail}")
                error_msg = error_detail.get('error', {}).get('message', str(e))
            except:
                error_msg = str(e)
            
            print(f"Error in Groq API request: {error_msg}")
            return f"Error generating report: {error_msg}"
            
        except requests.exceptions.Timeout:
            print("Timeout generating report")
            return "Error: The AI service took too long to respond. Please try again."
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Error generating report: {str(e)}"
    
    def run_full_research(self, query: str) -> Dict:
        """
        Execute complete research pipeline
        
        Args:
            query: User research question
            
        Returns:
            Dict with report, sources and statistics
        """
        print(f"\n{'='*60}")
        print(f"STARTING RESEARCH: {query}")
        print(f"{'='*60}\n")
        
        # 1. Search papers
        papers = self.fetch_arxiv_papers(query, max_results=15)
        
        if not papers:
            return {
                'status': 'error',
                'message': 'No papers found in arXiv',
                'report': None,
                'sources': []
            }
        
        # 2. Store in Qdrant
        self.store_in_qdrant(papers)
        
        # 3. Semantic search
        relevant_docs = self.search_relevant_docs(query, top_k=10)
        
        # 4. Generate report
        report = self.generate_report(query, relevant_docs)
        
        print(f"\n{'='*60}")
        print("RESEARCH COMPLETED")
        print(f"{'='*60}\n")
        
        return {
            'status': 'success',
            'report': report,
            'sources': relevant_docs[:8],  # Top 8 sources
            'total_papers': len(papers),
            'relevant_papers': len(relevant_docs)
        }