from media_api import MediaApi
import numpy as np
import configparser

if __name__ == '__main__':
    # Checking to see if things work

    url = 'https://api.ap.org/media'
    config = configparser.ConfigParser()
    config.read('password.ini')
    apikey = (config['key']['apikey'])
    media_api = MediaApi(url, apikey)

    # If you don't have any data files and want to make a request, do this
    # response = media_api.make_get_request()
    # print(response.status_code)
    # json_response = media_api.get_json_response()

    # If you want to read from a data file do this
    file_name = '../../data/media_1569096280.json'
    json_response = media_api.get_json_response_from_file(file_name)
    print(json_response)

    items = media_api.get_items()
    count = media_api.get_item_count()

    ind = np.random.randint(0, count-1)
    article_item = media_api.get_specific_item(ind) # the first item

    # getting specific things from the item
    print(article_item.get_headline())
    print(article_item.get_uri())
    print(article_item.get_full_json_response())
