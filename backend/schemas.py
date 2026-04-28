from pydantic import BaseModel

class OrderCreate(BaseModel):
    customer_name: str
    address: str