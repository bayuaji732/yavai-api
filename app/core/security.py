import jwt
import base64
from fastapi import HTTPException, Header
from app.core import config

def verify_token(token: str = Header(..., alias="Authorization")) -> dict:
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token is required")
    
    try:
        if token.startswith("Bearer "):
            token = token[7:]
        
        decoded = jwt.decode(
            token,
            base64.b64decode(config.JWT_TOKEN_KEY),
            algorithms=["HS256"],
            options={"verify": True}
        )
        return {
            "username": decoded.get("username"),
            "hadoop_username": decoded.get("hadoopUsername")
        }
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")