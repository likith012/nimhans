import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]import unittest

class TestSort(unittest.TestCase):
    
    def test_ascending_order(self):
        alist = ['1', '2', '3', '4', '5']
        sorted_list = sorted(alist, key=natural_keys)
        self.assertEqual(sorted_list, alist)
        
    def test_descending_order(self):
        alist = ['5', '4', '3', '2', '1']
        sorted_list = sorted(alist, key=natural_keys)
        self.assertEqual(sorted_list, alist)
        
    def test_mixed_order(self):
        alist = ['1', '2', '10', '20', '100', '200']
        sorted_list = sorted(alist, key=natural_keys)
        self.assertEqual(sorted_list, ['1', '2', '10', '20', '100', '200'])
        
if __name__ == '__main__':
    unittest.main()