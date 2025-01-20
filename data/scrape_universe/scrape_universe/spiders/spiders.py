from pathlib import Path

import yaml, json
import scrapy
import time
import tqdm


class ESOSpider(scrapy.Spider):
    name = "eso_spider"

    def start_requests(self):
        pages = 61
        for page in tqdm.range(1, pages + 1):
            url = f"https://www.eso.org/public/images/viewall/list/{page}/?&sort=-release_date"
            yield scrapy.Request(url=url, callback=self.parse_main_page)

    def parse_main_page(self, response):
        # We assume second script is the images one
        images = response.css("script::text")[2].get()
        images = yaml.safe_load(images[images.find("["):images.rfind("]") + 1])

        for image in images:
            time.sleep(1) # be nice to ESO
            yield response.follow(image["url"], callback=self.parse_image_page)

    def parse_image_page(self, response):
        image_div = response.css(".left-column")
        yield {
            "id": response.url.split('/')[-2],
            "title": image_div.css("h1::text").get(),
            "alttext": image_div.css("img::attr(alt)").get(),
            "explanation": ' '.join(image_div.css("p::text").getall()).strip(),
            "url": response.urljoin(image_div.css("img::attr(src)").get()),
        }


class HubbleSpider(scrapy.Spider):
    name = "hubble_spider"

    def start_requests(self):
        pages = 105 
        for page in tqdm.trange(1, pages + 1):
            url = f"https://esahubble.org/images/page/{page}/?sort=-release_date"
            yield scrapy.Request(url=url, callback=self.parse_main_page)

    def parse_main_page(self, response):
        images = response.css("script::text").get()
        images = yaml.safe_load(images[images.find("["):images.rfind("]") + 1])

        for image in images:
            time.sleep(1) # be nice to Hubble
            yield response.follow(image["url"], callback=self.parse_image_page)

    def parse_image_page(self, response):
        image_div = response.css(".left-column")
        yield {
            "id": response.url.split('/')[-2],
            "title": image_div.css("h1::text").get(),
            "alttext": image_div.css("img::attr(alt)").get(),
            "explanation": ' '.join(image_div.css("p::text").getall()).strip(),
            "url": response.urljoin(image_div.css("img::attr(src)").get()),
        }
