"""
Research Engine with Qdrant for semantic search
Prioritizes most recent papers from arXiv
"""
import os
import requests
from typing import List, Dict
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding


class ResearchEngine:
    def __init__(self):
        """Initialize research engine"""
        print("Initializing Research Engine...")
        
        # Validate Groq API Key
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not configured in .env")
        print("Groq API Key configured")
        
        # Qdrant Cloud Client
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
            print(f"Existing collections: {len(collections.collections)}")
            
        except Exception as e:
            print(f"Error connecting to Qdrant Cloud: {e}")
            raise
        
        # Embedding model with FastEmbed
        print("Loading embedding model with FastEmbed...")
        try:
            self.encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            print("FastEmbed model loaded")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self._setup_collection()
        print("Research Engine ready")
    
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
                    size=384,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created")
    
    def fetch_arxiv_papers(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Fetch recent papers from arXiv API
        Sorted by most recent submission date
        
        Args:
            query: Search term
            max_results: Maximum number of results
            
        Returns:
            List of papers with title, summary, URL, and publication date
        """
        print(f"Searching papers in arXiv: '{query}'")
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
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
                    summary = ' '.join(summary.split())[:800]
                    
                    link = entry.split('<id>')[1].split('</id>')[0].strip()
                    
                    # Extract publication date
                    published = entry.split('<published>')[1].split('</published>')[0].strip()
                    pub_date = published.split('T')[0]  # Format: YYYY-MM-DD
                    
                    papers.append({
                        'title': title,
                        'content': summary,
                        'url': link,
                        'source': 'arXiv',
                        'published_date': pub_date
                    })
                except:
                    continue
            
            # Sort by date (most recent first)
            papers.sort(key=lambda x: x['published_date'], reverse=True)
            
            print(f"Found {len(papers)} papers")
            if papers:
                print(f"Date range: {papers[-1]['published_date']} to {papers[0]['published_date']}")
            
            return papers
            
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def store_in_qdrant(self, documents: List[Dict]):
        """
        Store documents in Qdrant with embeddings
        Includes publication date in metadata
        
        Args:
            documents: List of documents with title, content, URL, and date
        """
        if not documents:
            print("No documents to store")
            return
        
        print(f"Storing {len(documents)} documents in Qdrant...")
        
        try:
            texts = [f"{doc['title']} {doc['content']}" for doc in documents]
            embeddings = list(self.encoder.embed(texts))
            
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                points.append(PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        'title': doc['title'],
                        'content': doc['content'],
                        'url': doc['url'],
                        'source': doc.get('source', 'unknown'),
                        'published_date': doc.get('published_date', 'unknown')
                    }
                ))
            
            self.qdrant.upsert(collection_name="research_docs", points=points)
            print(f"{len(points)} documents stored successfully")
            
        except Exception as e:
            print(f"Error storing in Qdrant: {e}")
    
    def search_relevant_docs(self, query: str, top_k: int = 12) -> List[Dict]:
        """
        Search relevant documents using semantic similarity
        Returns results sorted by combination of relevance and recency
        
        Args:
            query: User search query
            top_k: Number of results to return
            
        Returns:
            List of most relevant recent documents
        """
        print(f"Searching relevant documents for: '{query}'")
        
        try:
            query_embedding = list(self.encoder.embed([query]))[0]
            
            # Get more results to allow for recency filtering
            results = self.qdrant.search(
                collection_name="research_docs",
                query_vector=query_embedding.tolist(),
                limit=top_k * 2
            )
            
            relevant_docs = []
            for hit in results:
                relevant_docs.append({
                    'title': hit.payload['title'],
                    'content': hit.payload['content'],
                    'url': hit.payload['url'],
                    'source': hit.payload['source'],
                    'published_date': hit.payload.get('published_date', 'unknown'),
                    'relevance_score': hit.score
                })
            
            # Sort by date (most recent first) while maintaining relevance threshold
            relevant_docs = [doc for doc in relevant_docs if doc['relevance_score'] > 0.3]
            relevant_docs.sort(key=lambda x: x['published_date'], reverse=True)
            
            # Return top_k most recent papers
            final_docs = relevant_docs[:top_k]
            
            print(f"Found {len(final_docs)} relevant recent documents")
            if final_docs:
                print(f"Date range: {final_docs[-1]['published_date']} to {final_docs[0]['published_date']}")
            
            return final_docs
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def generate_report(self, query: str, documents: List[Dict]) -> str:
        """
        Generate research report using Groq API
        Highlights recency of papers in the report
        
        Args:
            query: Research question
            documents: List of relevant documents
            
        Returns:
            Report in text format
        """
        print("Generating report with Groq API...")
        
        # Prepare context with top 6 documents, emphasizing dates
        context_parts = []
        for i, doc in enumerate(documents[:6], 1):
            context_parts.append(
                f"[{i}] Title: {doc['title']}\n"
                f"Published: {doc['published_date']}\n"
                f"Abstract: {doc['content']}\n"
                f"URL: {doc['url']}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert research assistant specializing in academic literature review.

RESEARCH QUESTION:
{query}

RECENT ACADEMIC PAPERS (sorted by publication date, most recent first):
{context}

INSTRUCTIONS:
1. Generate a professional executive summary of maximum 450 words
2. Emphasize the MOST RECENT findings and developments
3. Identify key trends and patterns in recent research
4. Use references [1], [2], [3], etc. to cite sources
5. Mention publication dates when discussing findings to highlight recency
6. Write in clear, professional English
7. DO NOT invent information not present in the sources

REPORT:"""
        
        try:
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
                    "max_tokens": 1500,
                    "temperature": 0.3
                },
                timeout=60
            )
            
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
            return "Error: AI service took too long to respond. Please try again."
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Error generating report: {str(e)}"
    
    def run_full_research(self, query: str) -> Dict:
        """
        Execute complete research pipeline
        
        Args:
            query: User research question
            
        Returns:
            Dict with report, sources, and statistics
        """
        print(f"\n{'='*60}")
        print(f"STARTING RESEARCH: {query}")
        print(f"{'='*60}\n")
        
        # Fetch papers (increased to 20 for better recency filtering)
        papers = self.fetch_arxiv_papers(query, max_results=20)
        
        if not papers:
            return {
                'status': 'error',
                'message': 'No papers found in arXiv',
                'report': None,
                'sources': []
            }
        
        # Store in Qdrant
        self.store_in_qdrant(papers)
        
        # Semantic search with recency priority
        relevant = self.search_relevant_docs(query, top_k=10)
        
        # Generate report
        report = self.generate_report(query, relevant)
        
        print(f"\n{'='*60}")
        print("RESEARCH COMPLETED")
        print(f"{'='*60}\n")
        
        return {
            'status': 'success',
            'report': report,
            'sources': relevant[:8],
            'total_papers': len(papers),
            'relevant_papers': len(relevant)
        }