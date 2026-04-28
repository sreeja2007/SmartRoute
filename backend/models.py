from sqlalchemy import Column, Integer, String, Float
from database import Base

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    customer_name = Column(String)
    address = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    status = Column(String, default="pending")
    vehicle_id = Column(Integer, nullable=True)