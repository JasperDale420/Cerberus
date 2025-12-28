# unusual-whales-api-client
A client library for accessing Unusual Whales API

Thank you @unusualwhales for providing an excellent api!


## Example Implementation
```python
import os
import decimal
from dotenv import load_dotenv
from unusualwhales import UnusualWhalesClient, 

from unusualwhales.models import OffLitPriceLevelResults, OffLitPriceLevel
from unusualwhales.api.stock import get_volume_by_price_level

load_dotenv('.env')
UW_API_TOKEN = os.environ.get("UW_API_TOKEN", None)

def main():
    client = UnusualWhalesClient(base_url="https://api.unusualwhales.com", token=UW_API_TOKEN)
    with client as client:
        data: OffLitPriceLevelResults = get_volume_by_price_level.sync(client=client, ticker="SPY", date="2024-05-03")
        for row in data.data:
            rec: OffLitPriceLevel = row
            print(f"${rec.price}: Lit Vol:{decimal(str(rec.lit_vol))} | Dark Vol: {rec.off_vol}")

if __name__ == '__main__':
    main()
```

## API Support
This client supports all api endpoints with exception to websockets.

You can use the openapi.yaml file to generate your own clients for unusual whales in
any language openapi has a generator for (and alot exist).

## Async Support

This library works with async await to handle non blocking i/o for better performance!

```python
# same code as before but with async/await note that the function changes from sync_detailed to asyncio_detailed
# alternatively you can use sync and asyncio
from unusualwhales.client import UnusualWhalesClient

async with client as client:
    ticker_options_volume: TickerOptionsVolumeResults = await getTickerOptionsVolume.asyncio(client=client,ticker="AAPL",date="2024-05-03")
    for option_volume in ticker_options_volume.data:
        print(option_volume.strike, option_volume.volume)
```

Things to know:
1. Every path/method combo becomes a Python module with four functions:
    1. `sync`: Blocking request that returns parsed data (if successful) or `None`
    1. `sync_detailed`: Blocking request that always returns a `Request`, optionally with `parsed` set if the request was successful.
    1. `asyncio`: Like `sync` but async instead of blocking
    1. `asyncio_detailed`: Like `sync_detailed` but async instead of blocking

1. All path/query params, and bodies become method arguments.

## Advanced customizations

There are more settings on the generated `Client` class which let you control more runtime behavior, check out the docstring on that class for more info. You can also customize the underlying `httpx.Client` or `httpx.AsyncClient` (depending on your use-case):

```python
from unusual_whales_api_client import Client

def log_request(request):
    print(f"Request event hook: {request.method} {request.url} - Waiting for response")

def log_response(response):
    request = response.request
    print(f"Response event hook: {request.method} {request.url} - Status {response.status_code}")

client = Client(
    base_url="https://api.unusualwhales.com",
    token="YOUR_API_TOKEN"
    httpx_args={"event_hooks": {"request": [log_request], "response": [log_response]}},
)
```
