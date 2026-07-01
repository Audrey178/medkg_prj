import os


from qdrant_client import models as qdrant_models
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class VectorStore:
    def __init__(self, db_type):
        self.db_type = db_type
        
        if db_type == "qdrant":
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )
            
            self.ping()
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")
            
    def ping(self) -> tuple[bool, str]:
        try:
            if self.db_type == "qdrant":
                try:
                    response = self.client.from_("nonexistent_table_for_health_check").select("*").limit(1).execute()
                    logger.info("Successfully connected to Qdrant")
                    return True, "Successfully connected to Qdrant"
                except Exception as e:
                    logger.error(f"Error connecting to Qdrant: {e}")
                    return False, f"Error connecting to Qdrant: {e}"
        except Exception as e:
            logger.error(f"Error occurred while initializing vector store: {e}")
            return False, f"Error occurred while initializing vector store: {e}"
        
    
    def add_item(self, data: dict, collection_name: str):
        if self.db_type == "qdrant":
            # Prepare point data for Qdrant
            point_id = data.get("id")
            vector = data.get("vector")
            payload = data.get("payload", {})
            
            if point_id is None or vector is None:
                raise ValueError("[Qdrant] Data must contain 'id' and 'vector' fields.")
            
            collections = self.client.get_collections().collections
            collection_names = [ for col]