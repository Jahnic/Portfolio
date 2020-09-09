![](centris.jpeg)

# Introduction
While the reasons for homeownership may vary, asset appreciation is a common denominator. Those interested in buying a home will either need to spend a significant time on research or entrust a real estate agent to find the best fit. Because of the large volume and variety of relevant data, manual examinations will inevitably be limited. Many properties with high potential ROIs are likely to be missed. To address this issue, I propose the deployment of a regression model to estimate prices based on several indicators. Deviations from the expected value may reveal underappreciated real estate that can be used as starting points for investors and future homeowners.  

Data is retrieved through web scraping listings using `Selenium` on [Centris](https://www.centris.ca/en/properties~for-sale~montreal-island?view=Thumbnail).
Due to the complexity of the data retrieval, the process takes about 9 seconds per listing. It's therefore highly recommended to focus on a single city for the scraping process. An area with 6000 listings is estimated to take approximately 15 hours.

# Installation
1. Install required packages: `numpy`, `pandas`, `selenium`
2. Install [Chromedriver](http://chromedriver.chromium.org/). The specified installation path is `'C:/webdriver/chromedriver.exe'`. If you are not using windows you will need to change the path in the code.
3. Run `scrape.ipynb` to retrieve the data.
4. Run `wrangle.ipynb` to clean the data or write your own script. The wrangle script focuses on condos since it is the largest market in Montreal and the associated area features were most consistent. If interested in other housing types you will need to write your own scripts for wrangling and modeling the data.
