from pydantic import BaseModel

class request_body(BaseModel):
    customer_id : str
    timestamp : str
    states :   dict
    offers : dict

class request_put(BaseModel):
    customer_id: str
    timestamp : str
    accepted_one_of_the_three_offers: bool
