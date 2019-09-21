import requests
import json
import time
from article_item import ArticleItem

class MediaApi:
    '''
    A class for making requests to the media API and saving the responses as json
    '''
    def __init__(self, url, apikey, page_size=1):
        self.url = url
        self.apikey = apikey
        self.page_size=page_size

        # URL format: https://api.ap.org/media/v[{version}]/content/ondemand?apikey={apikey}[{optional_parameters}]
        self.full_url = '{}/content/ondemand?apikey={}'.format(url, apikey)
        self.json_response = None
        self.response = None
        self.file_name = None

    def make_get_request(self):
        '''
        Makes a get request. If the request is ok, a json file is saved with the response
        '''
        self.response = requests.get(self.full_url)
        # creating a unique file so we don't lose data
        self.file_name = '../../data/media_{}.json'.format(int(time.time()))

        # write to json file if the response is ok
        if self.response.status_code == 200:
            with open(self.file_name, 'x') as outfile:
                self.json_response = self.response.json()
                json.dump(self.json_response, outfile)
        return self.response

    def get_json_response(self):
        '''
        Call this function after making a request
        '''
        return self.json_response

    def get_json_response_from_file(self, file_name):
        '''
        Read json from a file
        '''
        with open(file_name, 'r') as infile:
            self.json_response = json.load(infile)
        return self.json_response

    def get_item_count(self):
        '''
        Gets the count of how many items there are in the json output
        '''
        return len(self.json_response['data']['items'])

    def get_items(self):
        '''
        Gets all items in the json response
        '''
        return self.json_response['data']['items']

    def get_specific_item(self, item_index):
        '''
        Return is of type ArticleItem
        '''
        if item_index < self.get_item_count():
            item_raw = self.json_response['data']['items'][item_index]['item']
            item = ArticleItem(self.apikey, item_raw)
            return item
        else:
            return None
