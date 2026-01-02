import regex
import jwt
import base64
from typing import Tuple
from core import config

def clean_table_name(table_name: str) -> str:
    """Clean and normalize table name"""
    table_name = table_name.lower()
    table_name = regex.sub(r'\s+', '_', table_name)
    table_name = regex.sub(r'[^A-Za-z0-9_]', '', table_name)
    return table_name

def clean_column_name(column_name: str) -> str:
    """Clean and normalize column name"""
    column_name = regex.sub(r'\s+', '_', column_name)
    column_name = regex.sub(r'[^A-Za-z0-9_]', '', column_name)
    return column_name.lower()

def parse_jwt_token(token: str) -> Tuple[str, str]:
    """Parse JWT token to extract username and hadoop username"""
    try:
        decoded = jwt.decode(
            token,
            base64.b64decode(config.JWT_TOKEN_KEY),
            algorithms=["HS256"],
            options={"verify": True}
        )
        return decoded.get("username"), decoded.get("hadoopUsername")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid JWT token")

def replace_boolean_string(text: str) -> str:
    """Replace JavaScript boolean strings with Python booleans"""
    text = regex.sub(r'\bfalse\b', 'False', text, flags=regex.IGNORECASE)
    text = regex.sub(r'\btrue\b', 'True', text, flags=regex.IGNORECASE)
    return text