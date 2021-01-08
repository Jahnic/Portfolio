import requests
from bs4 import BeautifulSoup
import pandas as pd
import random as rnd 
import time

'''
Extract links of listings
'''
# Request
base_url = "https://www.centris.ca/en/properties~for-sale~montreal-island?view=Thumbnail&uc="
payload={}
headers = {
    'Cookie': 'AnonymousId=af3f2205338d4de19bbeb6bc83d7be44;ll-search-selection=; .AspNetCore.Session=CfDJ8Pcb9O%2FqRVhEkAODRo0nLbUMkZhUeBGDafCRti1vjCyOCIbFtxVqN5D0LOfZXbDZqyu1nuYB8oX2DSnwVFqUs%2FFtSGE53l6SYprchaUmtAgtXHuvutAufQ7E0mCAtKhIV0vJpuM6DENtKM2T%2BgxmoLfMOqhZNcJgIEIZdmRvrtD2',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
}
# All listings
all_listings = []
# Iterate through pages of listing links (20 links per page)
for page in range(0, 301, 20):
    url = base_url + str(page)
    # Test new url
    try:
        response = requests.request("GET", url, headers=headers, data=payload)
        # Status code of request
        print("Response status code:", response.status_code)
    except: 
        print("Links scraped!")
        break
    
    # Parse text with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    # Get listing urls
    listings = [('https://www.centris.ca' + link.get("href")) 
                for link in soup.find_all(class_="a-more-detail")]
    
    # Append links to all_listings
    for link in listings:
        all_listings.append(link)
        
# Check for duplicates
print("No duplicates", len(all_listings) == len(set(all_listings)))

'''
Extract data on property
'''
for count, listing in enumerate(all_listings):
    print("Request listing: {}".format(count))
    response = requests.request("GET", listing, headers=headers, data=payload)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Parse data on properties
    lat = soup.find(itemprop='latitude').get('content')
    lon = soup.find(itemprop='longitude').get('content')
    price = soup.find(id='BuyPrice').text
    address = soup.find(class_='pt-1').text
    title = soup.select("div.col:nth-child(1) > h1:nth-child(1) > span:nth-child(1)")[0].get_text()
    property_data_dict = {
                        'title': title,
                        'lat': lat,
                        'long': lon,
                        'price': price,
                        'address': address
                        }

    # Data on property (number of rooms, net area, ...)
    property_data = [property_data.text for property_data 
                    in soup.find_all(class_="col-lg-3 col-sm-6 carac-container")]
    # Split string of property headers and data 
    property_data = [data.split('\n') for data in property_data]
    for data in property_data:
        # Remove '' from splitting
        data = data[1:-1]
        if data[0] != '':
            header = data[0]
            values = data[1:]
        else:
            # Only walkscores have no headers on webpage
            header = 'Walkscore'
            values = data[1:2]
        
        # Add data to dict
        if header in property_data_dict.keys():
            property_data_dict[header] = property_data_dict[header].append(values)
        else:
            property_data_dict[header] = [values]
        
# Tescht         
property_data_dict
pd.DataFrame(property_data_dict)

        