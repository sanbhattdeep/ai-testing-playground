# Import Scrapy's Spider base class.
#
# A Spider defines:
#
#   1. Which pages Scrapy should visit
#   2. What Scrapy should do with the responses
#
# Our custom spider below inherits from this class.
from scrapy import Spider


# ---------------------------------------------------------
# 1. DEFINE A CUSTOM SCRAPY SPIDER
# ---------------------------------------------------------

# Create our own spider by inheriting from Scrapy's
# built-in Spider class.
#
# Inheritance means EnduringAdmirationSpider gets the
# behavior provided by Spider, while we add our own:
#
#     name
#     start_urls
#     parse()
class EnduringAdmirationSpider(Spider):


    # -----------------------------------------------------
    # 2. GIVE THE SPIDER A NAME
    # -----------------------------------------------------

    # Every Scrapy spider needs a unique name.
    #
    # This is NOT necessarily the Python class name.
    #
    # Scrapy uses this value when you run the spider
    # from the command line.
    #
    # For example, inside a Scrapy project you could run:
    #
    #     scrapy crawl enduring_admiration_data
    #
    name = "enduring_admiration_data"


    # -----------------------------------------------------
    # 3. DEFINE THE STARTING URL
    # -----------------------------------------------------

    # start_urls is a list containing the URLs that Scrapy
    # should request when the spider begins running.
    #
    # Here there is only one URL.
    #
    # Conceptually:
    #
    # Spider starts
    #      ↓
    # Read start_urls
    #      ↓
    # Request this webpage
    #      ↓
    # Receive HTTP response
    #      ↓
    # Call parse(response)
    #
    start_urls = ["https://testerstories.com/files/ai_and_ml/ea-001.html"]


    # -----------------------------------------------------
    # 4. PROCESS THE HTTP RESPONSE
    # -----------------------------------------------------

    # Scrapy automatically calls parse() for responses
    # downloaded from start_urls.
    #
    # response represents the HTTP response returned
    # by the website.
    #
    # It contains information such as:
    #
    #     response.url
    #     response.status
    #     response.headers
    #     response.text
    #
    def parse(self, response):


        # -------------------------------------------------
        # 5. GET THE PAGE CONTENT
        # -------------------------------------------------

        # response.text contains the response body decoded
        # as text.
        #
        # IMPORTANT:
        #
        # For an HTML page, this means the HTML source,
        # not merely the human-visible words on the page.
        #
        # For example, response.text could contain:
        #
        #     <html>
        #       <body>
        #         <h1>Example</h1>
        #         <p>Hello world</p>
        #       </body>
        #     </html>
        #
        # So the variable "text" here actually contains
        # the complete HTML document returned by the server.
        text = response.text


        # -------------------------------------------------
        # 6. YIELD THE SCRAPED ITEM
        # -------------------------------------------------

        # yield sends a scraped item back to Scrapy.
        #
        # In this case the item is simply a Python dictionary:
        #
        #     {
        #         "text": "<html>...</html>"
        #     }
        #
        # Scrapy can then:
        #
        #     export it to JSON
        #     export it to CSV
        #     send it through an item pipeline
        #     store it somewhere else
        #
        # depending on how the Scrapy project is configured.
        yield {"text": text}