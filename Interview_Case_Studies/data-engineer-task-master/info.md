# Data Engineer Task
For the assignment two python scripts has been created `api.py` and `run.py`.

**api.py** contains the end point with URL for status and city. **run.py** is used to call the end point for processing the JSON file.

## Methodology

### API

We have used `FAST API` as service as FLASK is not compatible in my current web browser settings.

Processing for each end point is mentioned line to line in the form of code.
#### Endpoint user_status
`/user_id/{user_id}/date/{date}`

Thi is used to get status with the help of *date* and *user_id*.

#### Endpoint city
`/ip_city/{ip_city}`

On this endpoint please provide an implementation that searches the provided IP ranges and returns the correct city based on the IP.
### File Processing

The file `transactions.json` is read and the end points are called with the script which provides aggregated result containing the sum of `product_price` grouped by `user_status` and `city`.

## Setup

There is a simple API which you'll need to install.
To run the API just run the api.py file.

```
pip install -r requirements.txt
python api.py
```
