import re
import os
import sys
import numpy as np
from shutil import copyfile


min_ten_occurances=['text_text-body','text_reference-list-item', 'text_section-heading',
                        'text_page-number','text-with-special-symbols_text-body', 'text_caption',
                        'text_list-item', 'text_page-header', 'math_non-text', 'drawing_non-text',
                        'text-with-special-symbols_list-item', 'text_page-footer', 'ruling_non-text',
                        'halftone_non-text', 'tehttps://stackoverflow.com/questions/7419665/python-move-and-overwrite-files-and-foldersxt_author', 'text_abstract-body', 'text_title',
                        'text_footnote', 'table_non-text', 'text_affiliation', 'text_reference-heading',
                        'text_abstract-heading', 'text_not-clear', 'text-with-special-symbols_page-footer',
                        'text_biography', 'text-with-special-symbols_caption', 'text_keyword-body',
                        'text-with-special-symbols_page-header', 'text-with-special-symbols_reference-list-item',
                        'text_article-submission-information', 'text-with-special-symbols_footnote',
                        'halftone-with-drawing_non-text', 'text_list', 'text_keyword-heading',
                        'text-with-special-symbols_list', 'text_reference-list', 'text_drop-cap',
                        'text-with-special-symbols_abstract-body', 'text-with-special-symbols_author',
                        'text-with-special-symbols_affiliation', 'text_definition', 'text_membership',
                        'text_synopsis',  'text-with-special-symbols_pseudo-code', 'map_non-text',
                        'logo_non-text', 'text-with-special-symbols_reference-list',
                        'text-with-special-symbols_biography','text_keyword_heading_and_body',
                        'advertisement_non-text']

split=[]
p=re.compile("[a-zA-Z]+")
for f in min_ten_occurances:
    s=p.findall(f)
    split.append(s)
print str([x for x in split])
dir=sys.argv[1]
no_redundancy_data=os.path.join(dir,"10_or_more_data")
os.mkdir(no_redundancy_data,0777)
subdirs=os.listdir(dir)
low=0
high=0

for subdir in subdirs:
    subdir_p=os.path.join(dir,subdir)
    files=os.listdir(subdir_p)
    for file in files:
        fn=file.split(".")[0]
        fn=p.findall(fn)
        if fn  in split:
            file_p=os.path.join(subdir_p,file)
            dest_file=os.path.join(no_redundancy_data,file)
            if os.path.exists(dest_file):
                continue
            copyfile(file_p,dest_file)
            high=high+1
        else:
            low=low+1
print "high:"+str(high)
print "low:"+str(low)