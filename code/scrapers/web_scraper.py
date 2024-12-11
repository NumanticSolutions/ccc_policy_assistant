# Class for scraping websites
# Built using Pyppeteer

from pyppeteer import launch
from bs4 import BeautifulSoup
# import html2text
import urllib
import re

class webScraper:
    '''
    Class for scraping am individual website. Note that url is passed to the
    asynchronous visit_page method (see usage below) and (without errors) this
    method will return a dictionary with the raw html, a BeautifulSoup object, the
    text found in the HTML and the urls found in <a> tags.

    usage:
        turl = "https://en.wikipedia.org/wiki/Chaffey_College"
        test = await wc.webScraper.visit_page(url=turl)

        test.crawl_results.keys()
        dict_keys(['url', 'html_code_string', 'soup', 'soup_text', 'h2t_text', 'page_urls'])

    params:

    '''

    def __init__(self, crawl_results):
        '''
        Initialize class
        '''

        self.crawl_results = crawl_results

        # # Update any key word args
        # self.__dict__.update(kwargs)


    @classmethod
    async def visit_page(cls, url):
        '''
        Use Pyppeteeer to visit page and then return results in a dict
        :return:
        '''

        result = await cls.collect_page_data(url=url)
        # Create an instance
        instance = cls(crawl_results=result)
        return instance

    @staticmethod
    async def collect_page_data(url):
        '''
        Asynch function to get page content
        :return:
        '''

        try:
            browser = await launch()
            page = await browser.newPage()
            await page.goto(url=url)

            ## Step 1: Get page content from Pyppeter
            html = await page.content()
            await browser.close()

        except:
            return {}

        ## Step 2: Load HTML Response Into BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        ## Step 3: Get the HTML code in a string
        html_code_string = str(soup)

        ## Step 4: Get text from p tags
        ptag_texts = []
        for p in soup.find_all('p'):
            ptag_texts.append(p.text)

        ## Step 4b: Create a single clean text column
        # Remove unwanted characters
        pat = r"\n|\xa0"
        ptag_text = " ".join(ptag_texts)
        ptag_text = re.sub(pat, " ", ptag_text)
        pat = "\\s+"
        ptag_text = re.sub(pat, " ", ptag_text)

        # ## Step 5: Get the BeautifulSoup text
        # soup_text = soup.get_text()
        #
        # ## Step 6: Get the HTML2Text text
        # h = html2text.HTML2Text()
        # # Ignore converting links from HTML
        # h.ignore_links = True
        # h.body_width = 0
        # h2t_text = h.handle(html_code_string)

        # Return all links in <a tags with href values
        atag_urls = []
        for a in soup.find_all('a', href=True):
            atag_urls.append(a['href'])

        # Remove redundancies
        page_urls = set(atag_urls)

        # Append the base url to make it navigable
        purl = urllib.parse.urlparse(url)
        purl = "{}://{}".format(purl.scheme, purl.netloc)
        atag_urls = [urllib.parse.urljoin(purl, u) for u in page_urls if u.find("http") < 0]

        return dict(url=url,
                    html_code_string=html_code_string,
                    soup=soup,
                    ptag_text=ptag_text,
                    atag_urls=atag_urls)


