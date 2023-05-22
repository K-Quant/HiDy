The StrategyData contains industry policy data scraped from the website of the National Development and Reform Commission (NDRC). Using named entity recognition (NER), we can extract the following knowledge: (document_name, mention, industry, timestamp). 
The AnnualReportData includes annual report data of alcoholic beverage companies, with sections limited to "Board of Directors' Report", "Management Discussion and Analysis", etc. Through NER, we can extract the following knowledge: (company, produce, product, timestamp).

To obtain the results, you can run the NER.py directly.