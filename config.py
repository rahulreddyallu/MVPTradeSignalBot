"""
Configuration file for NIFTY 200 Trading Signal Bot
Contains all parameters, credentials and settings
"""

# Upstox API Credentials
UPSTOX_API_KEY = "ad55de1b-c7d1-4adc-b559-3830bf1efd72"
UPSTOX_API_SECRET = "969nyjgapm"
UPSTOX_REDIRECT_URI = "https://localhost"
UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI0TEFGUDkiLCJqdGkiOiI2N2UzNzhmZTMzMDFjNDE0MGRjZWY1NjMiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQyOTYwODk0LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDMwMjY0MDB9.LClpwKGUVL9j24NQvUDkMtx0yGDIPg9vvj4Fj5Ur7Mw"

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "7209852741:AAEf-_f6TeZK1-_R55yq365iU_54rk95y-c"
TELEGRAM_CHAT_ID = "936205208"
ENABLE_TELEGRAM_ALERTS = True

HISTORICAL_DAYS = 365

# Stock Universe
# List of NIFTY 200 stocks to analyze (instrument IDs from Upstox)
STOCK_LIST = [
    "NSE_EQ|INE117A01022", "NSE_EQ|INE012A01025", "NSE_EQ|INE702C01027", "NSE_EQ|INE949L01017", "NSE_EQ|INE931S01010", "NSE_EQ|INE423A01024", "NSE_EQ|INE364U01010", "NSE_EQ|INE742F01042", "NSE_EQ|INE814H01011", "NSE_EQ|INE399L01023", "NSE_EQ|INE674K01013", "NSE_EQ|INE647O01011", "NSE_EQ|INE540L01014", "NSE_EQ|INE079A01024", "NSE_EQ|INE437A01024", "NSE_EQ|INE438A01022", "NSE_EQ|INE208A01029", "NSE_EQ|INE021A01026", "NSE_EQ|INE006I01046", "NSE_EQ|INE406A01037", "NSE_EQ|INE192R01011", "NSE_EQ|INE238A01034", "NSE_EQ|INE118H01025", "NSE_EQ|INE917I01010", "NSE_EQ|INE296A01024", "NSE_EQ|INE918I01026", "NSE_EQ|INE118A01012", "NSE_EQ|INE787D01026", "NSE_EQ|INE545U01014", "NSE_EQ|INE028A01039", "NSE_EQ|INE084A01016", "NSE_EQ|INE457A01014", "NSE_EQ|INE171Z01026", "NSE_EQ|INE263A01024", "NSE_EQ|INE465A01025", "NSE_EQ|INE257A01026", "NSE_EQ|INE029A01011", "NSE_EQ|INE397D01024", "NSE_EQ|INE343G01021", "NSE_EQ|INE376G01013", "NSE_EQ|INE323A01026", "NSE_EQ|INE216A01030", "NSE_EQ|INE067A01029", "NSE_EQ|INE476A01022", "NSE_EQ|INE121A01024", "NSE_EQ|INE059A01026", "NSE_EQ|INE522F01014", "NSE_EQ|INE704P01025", "NSE_EQ|INE591G01017", "NSE_EQ|INE259A01022", "NSE_EQ|INE111A01025", "NSE_EQ|INE298A01020", "NSE_EQ|INE271C01023", "NSE_EQ|INE016A01026", "NSE_EQ|INE148O01028", "NSE_EQ|INE361B01024", "NSE_EQ|INE935N01020", "NSE_EQ|INE089A01031", "NSE_EQ|INE066A01021", "NSE_EQ|INE042A01014", "NSE_EQ|INE302A01020", "NSE_EQ|INE388Y01029", "NSE_EQ|INE171A01029", "NSE_EQ|INE188A01015", "NSE_EQ|INE129A01019", "NSE_EQ|INE776C01039", "NSE_EQ|INE102D01028", "NSE_EQ|INE484J01027", "NSE_EQ|INE047A01021", "NSE_EQ|INE860A01027", "NSE_EQ|INE127D01025", "NSE_EQ|INE040A01034", "NSE_EQ|INE795G01014", "NSE_EQ|INE176B01034", "NSE_EQ|INE158A01026", "NSE_EQ|INE038A01020", "NSE_EQ|INE066F01020", "NSE_EQ|INE094A01015", "NSE_EQ|INE030A01027", "NSE_EQ|INE267A01025", "NSE_EQ|INE031A01017", "NSE_EQ|INE090A01021", "NSE_EQ|INE765G01017", "NSE_EQ|INE726G01019", "NSE_EQ|INE008A01015", "NSE_EQ|INE092T01019", "NSE_EQ|INE821I01022", "NSE_EQ|INE154A01025", "NSE_EQ|INE562A01011", "NSE_EQ|INE053A01029", "NSE_EQ|INE242A01010", "NSE_EQ|INE565A01014", "NSE_EQ|INE335Y01020", "NSE_EQ|INE053F01010", "NSE_EQ|INE202E01016", "NSE_EQ|INE203G01027", "NSE_EQ|INE121J01017", "NSE_EQ|INE095A01012", "NSE_EQ|INE663F01024", "NSE_EQ|INE009A01021", "NSE_EQ|INE646L01027", "NSE_EQ|INE121E01018", "NSE_EQ|INE880J01026", "NSE_EQ|INE019A01038", "NSE_EQ|INE749A01030", "NSE_EQ|INE758E01017", "NSE_EQ|INE797F01020", "NSE_EQ|INE04I401011", "NSE_EQ|INE303R01014", "NSE_EQ|INE237A01028", "NSE_EQ|INE498L01015", "NSE_EQ|INE115A01026", "NSE_EQ|INE214T01019", "NSE_EQ|INE018A01030", "NSE_EQ|INE0J1Y01017", "NSE_EQ|INE326A01037", "NSE_EQ|INE883A01011", "NSE_EQ|INE670K01029", "NSE_EQ|INE774D01024", "NSE_EQ|INE101A01026", "NSE_EQ|INE103A01014", "NSE_EQ|INE634S01028", "NSE_EQ|INE196A01026", "NSE_EQ|INE585B01010", "NSE_EQ|INE180A01020", "NSE_EQ|INE027H01010", "NSE_EQ|INE249Z01020", "NSE_EQ|INE356A01018", "NSE_EQ|INE414G01012", "NSE_EQ|INE848E01016", "NSE_EQ|INE589A01014", "NSE_EQ|INE584A01023", "NSE_EQ|INE733E01010", "NSE_EQ|INE239A01024", "NSE_EQ|INE093I01010", "NSE_EQ|INE213A01029", "NSE_EQ|INE274J01014", "NSE_EQ|INE982J01020", "NSE_EQ|INE881D01027", "NSE_EQ|INE417T01026", "NSE_EQ|INE603J01030", "NSE_EQ|INE761H01022", "NSE_EQ|INE619A01035", "NSE_EQ|INE262H01021", "NSE_EQ|INE347G01014", "NSE_EQ|INE211B01039", "NSE_EQ|INE318A01026", "NSE_EQ|INE455K01017", "NSE_EQ|INE511C01022", "NSE_EQ|INE134E01011", "NSE_EQ|INE752E01010", "NSE_EQ|INE811K01011", "NSE_EQ|INE160A01022", "NSE_EQ|INE020B01018", "NSE_EQ|INE415G01027", "NSE_EQ|INE002A01018", "NSE_EQ|INE018E01016", "NSE_EQ|INE123W01016", "NSE_EQ|INE002L01015", "NSE_EQ|INE647A01010", "NSE_EQ|INE775A01035", "NSE_EQ|INE070A01015", "NSE_EQ|INE721A01047", "NSE_EQ|INE003A01024", "NSE_EQ|INE343H01029", "NSE_EQ|INE073K01018", "NSE_EQ|INE062A01020", "NSE_EQ|INE114A01011", "NSE_EQ|INE044A01036", "NSE_EQ|INE660A01013", "NSE_EQ|INE195A01028", "NSE_EQ|INE040H01021", "NSE_EQ|INE494B01023", "NSE_EQ|INE092A01019", "NSE_EQ|INE151A01013", "NSE_EQ|INE467B01029", "NSE_EQ|INE192A01025", "NSE_EQ|INE670A01012", "NSE_EQ|INE155A01022", "NSE_EQ|INE245A01021", "NSE_EQ|INE081A01020", "NSE_EQ|INE142M01025", "NSE_EQ|INE669C01036", "NSE_EQ|INE280A01028", "NSE_EQ|INE685A01028", "NSE_EQ|INE813H01021", "NSE_EQ|INE849A01020", "NSE_EQ|INE974X01010", "NSE_EQ|INE628A01036", "NSE_EQ|INE481G01011", "NSE_EQ|INE692A01016", "NSE_EQ|INE854D01024", "NSE_EQ|INE200M01039", "NSE_EQ|INE205A01025", "NSE_EQ|INE669E01016", "NSE_EQ|INE226A01021", "NSE_EQ|INE075A01022", "NSE_EQ|INE528G01035", "NSE_EQ|INE758T01015", "NSE_EQ|INE010B01027"
]

# Complete Stock information mapping for all 200 stocks
STOCK_INFO = {
    "INE117A01022": {"name": "ABB India Ltd.", "industry": "Capital Goods", "symbol": "ABB", "series": "EQ"},
    "INE012A01025": {"name": "ACC Ltd.", "industry": "Construction Materials", "symbol": "ACC", "series": "EQ"},
    "INE702C01027": {"name": "APL Apollo Tubes Ltd.", "industry": "Capital Goods", "symbol": "APLAPOLLO", "series": "EQ"},
    "INE949L01017": {"name": "AU Small Finance Bank Ltd.", "industry": "Financial Services", "symbol": "AUBANK", "series": "EQ"},
    "INE931S01010": {"name": "Adani Energy Solutions Ltd.", "industry": "Power", "symbol": "ADANIENSOL", "series": "EQ"},
    "INE423A01024": {"name": "Adani Enterprises Ltd.", "industry": "Metals & Mining", "symbol": "ADANIENT", "series": "EQ"},
    "INE364U01010": {"name": "Adani Green Energy Ltd.", "industry": "Power", "symbol": "ADANIGREEN", "series": "EQ"},
    "INE742F01042": {"name": "Adani Ports and Special Economic Zone Ltd.", "industry": "Services", "symbol": "ADANIPORTS", "series": "EQ"},
    "INE814H01011": {"name": "Adani Power Ltd.", "industry": "Power", "symbol": "ADANIPOWER", "series": "EQ"},
    "INE399L01023": {"name": "Adani Total Gas Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "ATGL", "series": "EQ"},
    "INE674K01013": {"name": "Aditya Birla Capital Ltd.", "industry": "Financial Services", "symbol": "ABCAPITAL", "series": "EQ"},
    "INE647O01011": {"name": "Aditya Birla Fashion and Retail Ltd.", "industry": "Consumer Services", "symbol": "ABFRL", "series": "EQ"},
    "INE540L01014": {"name": "Alkem Laboratories Ltd.", "industry": "Healthcare", "symbol": "ALKEM", "series": "EQ"},
    "INE079A01024": {"name": "Ambuja Cements Ltd.", "industry": "Construction Materials", "symbol": "AMBUJACEM", "series": "EQ"},
    "INE437A01024": {"name": "Apollo Hospitals Enterprise Ltd.", "industry": "Healthcare", "symbol": "APOLLOHOSP", "series": "EQ"},
    "INE438A01022": {"name": "Apollo Tyres Ltd.", "industry": "Automobile and Auto Components", "symbol": "APOLLOTYRE", "series": "EQ"},
    "INE208A01029": {"name": "Ashok Leyland Ltd.", "industry": "Capital Goods", "symbol": "ASHOKLEY", "series": "EQ"},
    "INE021A01026": {"name": "Asian Paints Ltd.", "industry": "Consumer Durables", "symbol": "ASIANPAINT", "series": "EQ"},
    "INE006I01046": {"name": "Astral Ltd.", "industry": "Capital Goods", "symbol": "ASTRAL", "series": "EQ"},
    "INE406A01037": {"name": "Aurobindo Pharma Ltd.", "industry": "Healthcare", "symbol": "AUROPHARMA", "series": "EQ"},
    "INE192R01011": {"name": "Avenue Supermarts Ltd.", "industry": "Consumer Services", "symbol": "DMART", "series": "EQ"},
    "INE238A01034": {"name": "Axis Bank Ltd.", "industry": "Financial Services", "symbol": "AXISBANK", "series": "EQ"},
    "INE118H01025": {"name": "BSE Ltd.", "industry": "Financial Services", "symbol": "BSE", "series": "EQ"},
    "INE917I01010": {"name": "Bajaj Auto Ltd.", "industry": "Automobile and Auto Components", "symbol": "BAJAJ-AUTO", "series": "EQ"},
    "INE296A01024": {"name": "Bajaj Finance Ltd.", "industry": "Financial Services", "symbol": "BAJFINANCE", "series": "EQ"},
    "INE918I01026": {"name": "Bajaj Finserv Ltd.", "industry": "Financial Services", "symbol": "BAJAJFINSV", "series": "EQ"},
    "INE118A01012": {"name": "Bajaj Holdings & Investment Ltd.", "industry": "Financial Services", "symbol": "BAJAJHLDNG", "series": "EQ"},
    "INE787D01026": {"name": "Balkrishna Industries Ltd.", "industry": "Automobile and Auto Components", "symbol": "BALKRISIND", "series": "EQ"},
    "INE545U01014": {"name": "Bandhan Bank Ltd.", "industry": "Financial Services", "symbol": "BANDHANBNK", "series": "EQ"},
    "INE028A01039": {"name": "Bank of Baroda", "industry": "Financial Services", "symbol": "BANKBARODA", "series": "EQ"},
    "INE084A01016": {"name": "Bank of India", "industry": "Financial Services", "symbol": "BANKINDIA", "series": "EQ"},
    "INE457A01014": {"name": "Bank of Maharashtra", "industry": "Financial Services", "symbol": "MAHABANK", "series": "EQ"},
    "INE171Z01026": {"name": "Bharat Dynamics Ltd.", "industry": "Capital Goods", "symbol": "BDL", "series": "EQ"},
    "INE263A01024": {"name": "Bharat Electronics Ltd.", "industry": "Capital Goods", "symbol": "BEL", "series": "EQ"},
    "INE465A01025": {"name": "Bharat Forge Ltd.", "industry": "Automobile and Auto Components", "symbol": "BHARATFORG", "series": "EQ"},
    "INE257A01026": {"name": "Bharat Heavy Electricals Ltd.", "industry": "Capital Goods", "symbol": "BHEL", "series": "EQ"},
    "INE029A01011": {"name": "Bharat Petroleum Corporation Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "BPCL", "series": "EQ"},
    "INE397D01024": {"name": "Bharti Airtel Ltd.", "industry": "Telecommunication", "symbol": "BHARTIARTL", "series": "EQ"},
    "INE343G01021": {"name": "Bharti Hexacom Ltd.", "industry": "Telecommunication", "symbol": "BHARTIHEXA", "series": "EQ"},
    "INE376G01013": {"name": "Biocon Ltd.", "industry": "Healthcare", "symbol": "BIOCON", "series": "EQ"},
    "INE323A01026": {"name": "Bosch Ltd.", "industry": "Automobile and Auto Components", "symbol": "BOSCHLTD", "series": "EQ"},
    "INE216A01030": {"name": "Britannia Industries Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "BRITANNIA", "series": "EQ"},
    "INE067A01029": {"name": "CG Power and Industrial Solutions Ltd.", "industry": "Capital Goods", "symbol": "CGPOWER", "series": "EQ"},
    "INE476A01022": {"name": "Canara Bank", "industry": "Financial Services", "symbol": "CANBK", "series": "EQ"},
    "INE121A01024": {"name": "Cholamandalam Investment and Finance Company Ltd.", "industry": "Financial Services", "symbol": "CHOLAFIN", "series": "EQ"},
    "INE059A01026": {"name": "Cipla Ltd.", "industry": "Healthcare", "symbol": "CIPLA", "series": "EQ"},
    "INE522F01014": {"name": "Coal India Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "COALINDIA", "series": "EQ"},
    "INE704P01025": {"name": "Cochin Shipyard Ltd.", "industry": "Capital Goods", "symbol": "COCHINSHIP", "series": "EQ"},
    "INE591G01017": {"name": "Coforge Ltd.", "industry": "Information Technology", "symbol": "COFORGE", "series": "EQ"},
    "INE259A01022": {"name": "Colgate Palmolive (India) Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "COLPAL", "series": "EQ"},
    "INE111A01025": {"name": "Container Corporation of India Ltd.", "industry": "Services", "symbol": "CONCOR", "series": "EQ"},
    "INE298A01020": {"name": "Cummins India Ltd.", "industry": "Capital Goods", "symbol": "CUMMINSIND", "series": "EQ"},
    "INE271C01023": {"name": "DLF Ltd.", "industry": "Realty", "symbol": "DLF", "series": "EQ"},
    "INE016A01026": {"name": "Dabur India Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "DABUR", "series": "EQ"},
    "INE148O01028": {"name": "Delhivery Ltd.", "industry": "Services", "symbol": "DELHIVERY", "series": "EQ"},
    "INE361B01024": {"name": "Divi's Laboratories Ltd.", "industry": "Healthcare", "symbol": "DIVISLAB", "series": "EQ"},
    "INE935N01020": {"name": "Dixon Technologies (India) Ltd.", "industry": "Consumer Durables", "symbol": "DIXON", "series": "EQ"},
    "INE089A01031": {"name": "Dr. Reddy's Laboratories Ltd.", "industry": "Healthcare", "symbol": "DRREDDY", "series": "EQ"},
    "INE066A01021": {"name": "Eicher Motors Ltd.", "industry": "Automobile and Auto Components", "symbol": "EICHERMOT", "series": "EQ"},
    "INE042A01014": {"name": "Escorts Kubota Ltd.", "industry": "Capital Goods", "symbol": "ESCORTS", "series": "EQ"},
    "INE302A01020": {"name": "Exide Industries Ltd.", "industry": "Automobile and Auto Components", "symbol": "EXIDEIND", "series": "EQ"},
    "INE388Y01029": {"name": "FSN E-Commerce Ventures Ltd.", "industry": "Consumer Services", "symbol": "NYKAA", "series": "EQ"},
    "INE171A01029": {"name": "Federal Bank Ltd.", "industry": "Financial Services", "symbol": "FEDERALBNK", "series": "EQ"},
    "INE188A01015": {"name": "Fertilisers and Chemicals Travancore Ltd.", "industry": "Chemicals", "symbol": "FACT", "series": "EQ"},
    "INE129A01019": {"name": "GAIL (India) Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "GAIL", "series": "EQ"},
    "INE776C01039": {"name": "GMR Airports Ltd.", "industry": "Services", "symbol": "GMRAIRPORT", "series": "EQ"},
    "INE102D01028": {"name": "Godrej Consumer Products Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "GODREJCP", "series": "EQ"},
    "INE484J01027": {"name": "Godrej Properties Ltd.", "industry": "Realty", "symbol": "GODREJPROP", "series": "EQ"},
    "INE047A01021": {"name": "Grasim Industries Ltd.", "industry": "Construction Materials", "symbol": "GRASIM", "series": "EQ"},
    "INE860A01027": {"name": "HCL Technologies Ltd.", "industry": "Information Technology", "symbol": "HCLTECH", "series": "EQ"},
    "INE127D01025": {"name": "HDFC Asset Management Company Ltd.", "industry": "Financial Services", "symbol": "HDFCAMC", "series": "EQ"},
    "INE040A01034": {"name": "HDFC Bank Ltd.", "industry": "Financial Services", "symbol": "HDFCBANK", "series": "EQ"},
    "INE795G01014": {"name": "HDFC Life Insurance Company Ltd.", "industry": "Financial Services", "symbol": "HDFCLIFE", "series": "EQ"},
    "INE176B01034": {"name": "Havells India Ltd.", "industry": "Consumer Durables", "symbol": "HAVELLS", "series": "EQ"},
    "INE158A01026": {"name": "Hero MotoCorp Ltd.", "industry": "Automobile and Auto Components", "symbol": "HEROMOTOCO", "series": "EQ"},
    "INE038A01020": {"name": "Hindalco Industries Ltd.", "industry": "Metals & Mining", "symbol": "HINDALCO", "series": "EQ"},
    "INE066F01020": {"name": "Hindustan Aeronautics Ltd.", "industry": "Capital Goods", "symbol": "HAL", "series": "EQ"},
    "INE094A01015": {"name": "Hindustan Petroleum Corporation Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "HINDPETRO", "series": "EQ"},
    "INE030A01027": {"name": "Hindustan Unilever Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "HINDUNILVR", "series": "EQ"},
    "INE267A01025": {"name": "Hindustan Zinc Ltd.", "industry": "Metals & Mining", "symbol": "HINDZINC", "series": "EQ"},
    "INE031A01017": {"name": "Housing & Urban Development Corporation Ltd.", "industry": "Financial Services", "symbol": "HUDCO", "series": "EQ"},
    "INE090A01021": {"name": "ICICI Bank Ltd.", "industry": "Financial Services", "symbol": "ICICIBANK", "series": "EQ"},
    "INE765G01017": {"name": "ICICI Lombard General Insurance Company Ltd.", "industry": "Financial Services", "symbol": "ICICIGI", "series": "EQ"},
    "INE726G01019": {"name": "ICICI Prudential Life Insurance Company Ltd.", "industry": "Financial Services", "symbol": "ICICIPRULI", "series": "EQ"},
    "INE008A01015": {"name": "IDBI Bank Ltd.", "industry": "Financial Services", "symbol": "IDBI", "series": "EQ"},
    "INE092T01019": {"name": "IDFC First Bank Ltd.", "industry": "Financial Services", "symbol": "IDFCFIRSTB", "series": "EQ"},
    "INE821I01022": {"name": "IRB Infrastructure Developers Ltd.", "industry": "Construction", "symbol": "IRB", "series": "EQ"},
    "INE154A01025": {"name": "ITC Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "ITC", "series": "EQ"},
    "INE562A01011": {"name": "Indian Bank", "industry": "Financial Services", "symbol": "INDIANB", "series": "EQ"},
    "INE053A01029": {"name": "Indian Hotels Co. Ltd.", "industry": "Consumer Services", "symbol": "INDHOTEL", "series": "EQ"},
    "INE242A01010": {"name": "Indian Oil Corporation Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "IOC", "series": "EQ"},
    "INE565A01014": {"name": "Indian Overseas Bank", "industry": "Financial Services", "symbol": "IOB", "series": "EQ"},
    "INE335Y01020": {"name": "Indian Railway Catering And Tourism Corporation Ltd.", "industry": "Consumer Services", "symbol": "IRCTC", "series": "EQ"},
    "INE053F01010": {"name": "Indian Railway Finance Corporation Ltd.", "industry": "Financial Services", "symbol": "IRFC", "series": "EQ"},
    "INE202E01016": {"name": "Indian Renewable Energy Development Agency Ltd.", "industry": "Financial Services", "symbol": "IREDA", "series": "EQ"},
    "INE203G01027": {"name": "Indraprastha Gas Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "IGL", "series": "EQ"},
    "INE121J01017": {"name": "Indus Towers Ltd.", "industry": "Telecommunication", "symbol": "INDUSTOWER", "series": "EQ"},
    "INE095A01012": {"name": "IndusInd Bank Ltd.", "industry": "Financial Services", "symbol": "INDUSINDBK", "series": "EQ"},
    "INE663F01024": {"name": "Info Edge (India) Ltd.", "industry": "Consumer Services", "symbol": "NAUKRI", "series": "EQ"},
    "INE009A01021": {"name": "Infosys Ltd.", "industry": "Information Technology", "symbol": "INFY", "series": "EQ"},
    "INE646L01027": {"name": "InterGlobe Aviation Ltd.", "industry": "Services", "symbol": "INDIGO", "series": "EQ"},
    "INE121E01018": {"name": "JSW Energy Ltd.", "industry": "Power", "symbol": "JSWENERGY", "series": "EQ"},
    "INE880J01026": {"name": "JSW Infrastructure Ltd.", "industry": "Services", "symbol": "JSWINFRA", "series": "EQ"},
    "INE019A01038": {"name": "JSW Steel Ltd.", "industry": "Metals & Mining", "symbol": "JSWSTEEL", "series": "EQ"},
    "INE749A01030": {"name": "Jindal Steel & Power Ltd.", "industry": "Metals & Mining", "symbol": "JINDALSTEL", "series": "EQ"},
    "INE758E01017": {"name": "Jio Financial Services Ltd.", "industry": "Financial Services", "symbol": "JIOFIN", "series": "EQ"},
    "INE797F01020": {"name": "Jubilant Foodworks Ltd.", "industry": "Consumer Services", "symbol": "JUBLFOOD", "series": "EQ"},
    "INE04I401011": {"name": "KPIT Technologies Ltd.", "industry": "Information Technology", "symbol": "KPITTECH", "series": "EQ"},
    "INE303R01014": {"name": "Kalyan Jewellers India Ltd.", "industry": "Consumer Durables", "symbol": "KALYANKJIL", "series": "EQ"},
    "INE237A01028": {"name": "Kotak Mahindra Bank Ltd.", "industry": "Financial Services", "symbol": "KOTAKBANK", "series": "EQ"},
    "INE498L01015": {"name": "L&T Finance Ltd.", "industry": "Financial Services", "symbol": "LTF", "series": "EQ"},
    "INE115A01026": {"name": "LIC Housing Finance Ltd.", "industry": "Financial Services", "symbol": "LICHSGFIN", "series": "EQ"},
    "INE214T01019": {"name": "LTIMindtree Ltd.", "industry": "Information Technology", "symbol": "LTIM", "series": "EQ"},
    "INE018A01030": {"name": "Larsen & Toubro Ltd.", "industry": "Construction", "symbol": "LT", "series": "EQ"},
    "INE0J1Y01017": {"name": "Life Insurance Corporation of India", "industry": "Financial Services", "symbol": "LICI", "series": "EQ"},
    "INE326A01037": {"name": "Lupin Ltd.", "industry": "Healthcare", "symbol": "LUPIN", "series": "EQ"},
    "INE883A01011": {"name": "MRF Ltd.", "industry": "Automobile and Auto Components", "symbol": "MRF", "series": "EQ"},
    "INE670K01029": {"name": "Macrotech Developers Ltd.", "industry": "Realty", "symbol": "LODHA", "series": "EQ"},
    "INE774D01024": {"name": "Mahindra & Mahindra Financial Services Ltd.", "industry": "Financial Services", "symbol": "M&MFIN", "series": "EQ"},
    "INE101A01026": {"name": "Mahindra & Mahindra Ltd.", "industry": "Automobile and Auto Components", "symbol": "M&M", "series": "EQ"},
    "INE103A01014": {"name": "Mangalore Refinery & Petrochemicals Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "MRPL", "series": "EQ"},
    "INE634S01028": {"name": "Mankind Pharma Ltd.", "industry": "Healthcare", "symbol": "MANKIND", "series": "EQ"},
    "INE196A01026": {"name": "Marico Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "MARICO", "series": "EQ"},
    "INE585B01010": {"name": "Maruti Suzuki India Ltd.", "industry": "Automobile and Auto Components", "symbol": "MARUTI", "series": "EQ"},
    "INE180A01020": {"name": "Max Financial Services Ltd.", "industry": "Financial Services", "symbol": "MFSL", "series": "EQ"},
    "INE027H01010": {"name": "Max Healthcare Institute Ltd.", "industry": "Healthcare", "symbol": "MAXHEALTH", "series": "EQ"},
    "INE249Z01020": {"name": "Mazagoan Dock Shipbuilders Ltd.", "industry": "Capital Goods", "symbol": "MAZDOCK", "series": "EQ"},
    "INE356A01018": {"name": "MphasiS Ltd.", "industry": "Information Technology", "symbol": "MPHASIS", "series": "EQ"},
    "INE414G01012": {"name": "Muthoot Finance Ltd.", "industry": "Financial Services", "symbol": "MUTHOOTFIN", "series": "EQ"},
    "INE848E01016": {"name": "NHPC Ltd.", "industry": "Power", "symbol": "NHPC", "series": "EQ"},
    "INE589A01014": {"name": "NLC India Ltd.", "industry": "Power", "symbol": "NLCINDIA", "series": "EQ"},
    "INE584A01023": {"name": "NMDC Ltd.", "industry": "Metals & Mining", "symbol": "NMDC", "series": "EQ"},
    "INE733E01010": {"name": "NTPC Ltd.", "industry": "Power", "symbol": "NTPC", "series": "EQ"},
    "INE239A01024": {"name": "Nestle India Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "NESTLEIND", "series": "EQ"},
    "INE093I01010": {"name": "Oberoi Realty Ltd.", "industry": "Realty", "symbol": "OBEROIRLTY", "series": "EQ"},
    "INE213A01029": {"name": "Oil & Natural Gas Corporation Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "ONGC", "series": "EQ"},
    "INE274J01014": {"name": "Oil India Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "OIL", "series": "EQ"},
    "INE982J01020": {"name": "One 97 Communications Ltd.", "industry": "Financial Services", "symbol": "PAYTM", "series": "EQ"},
    "INE881D01027": {"name": "Oracle Financial Services Software Ltd.", "industry": "Information Technology", "symbol": "OFSS", "series": "EQ"},
    "INE417T01026": {"name": "PB Fintech Ltd.", "industry": "Financial Services", "symbol": "POLICYBZR", "series": "EQ"},
    "INE603J01030": {"name": "PI Industries Ltd.", "industry": "Chemicals", "symbol": "PIIND", "series": "EQ"},
    "INE761H01022": {"name": "Page Industries Ltd.", "industry": "Textiles", "symbol": "PAGEIND", "series": "EQ"},
    "INE619A01035": {"name": "Patanjali Foods Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "PATANJALI", "series": "EQ"},
    "INE262H01021": {"name": "Persistent Systems Ltd.", "industry": "Information Technology", "symbol": "PERSISTENT", "series": "EQ"},
    "INE347G01014": {"name": "Petronet LNG Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "PETRONET", "series": "EQ"},
    "INE211B01039": {"name": "Phoenix Mills Ltd.", "industry": "Realty", "symbol": "PHOENIXLTD", "series": "EQ"},
    "INE318A01026": {"name": "Pidilite Industries Ltd.", "industry": "Chemicals", "symbol": "PIDILITIND", "series": "EQ"},
    "INE455K01017": {"name": "Polycab India Ltd.", "industry": "Capital Goods", "symbol": "POLYCAB", "series": "EQ"},
    "INE511C01022": {"name": "Poonawalla Fincorp Ltd.", "industry": "Financial Services", "symbol": "POONAWALLA", "series": "EQ"},
    "INE134E01011": {"name": "Power Finance Corporation Ltd.", "industry": "Financial Services", "symbol": "PFC", "series": "EQ"},
    "INE752E01010": {"name": "Power Grid Corporation of India Ltd.", "industry": "Power", "symbol": "POWERGRID", "series": "EQ"},
    "INE811K01011": {"name": "Prestige Estates Projects Ltd.", "industry": "Realty", "symbol": "PRESTIGE", "series": "EQ"},
    "INE160A01022": {"name": "Punjab National Bank", "industry": "Financial Services", "symbol": "PNB", "series": "EQ"},
    "INE020B01018": {"name": "REC Ltd.", "industry": "Financial Services", "symbol": "RECLTD", "series": "EQ"},
    "INE415G01027": {"name": "Rail Vikas Nigam Ltd.", "industry": "Construction", "symbol": "RVNL", "series": "EQ"},
    "INE002A01018": {"name": "Reliance Industries Ltd.", "industry": "Oil Gas & Consumable Fuels", "symbol": "RELIANCE", "series": "EQ"},
    "INE018E01016": {"name": "SBI Cards and Payment Services Ltd.", "industry": "Financial Services", "symbol": "SBICARD", "series": "EQ"},
    "INE123W01016": {"name": "SBI Life Insurance Company Ltd.", "industry": "Financial Services", "symbol": "SBILIFE", "series": "EQ"},
    "INE002L01015": {"name": "SJVN Ltd.", "industry": "Power", "symbol": "SJVN", "series": "EQ"},
    "INE647A01010": {"name": "SRF Ltd.", "industry": "Chemicals", "symbol": "SRF", "series": "EQ"},
    "INE775A01035": {"name": "Samvardhana Motherson International Ltd.", "industry": "Automobile and Auto Components", "symbol": "MOTHERSON", "series": "EQ"},
    "INE070A01015": {"name": "Shree Cement Ltd.", "industry": "Construction Materials", "symbol": "SHREECEM", "series": "EQ"},
    "INE721A01047": {"name": "Shriram Finance Ltd.", "industry": "Financial Services", "symbol": "SHRIRAMFIN", "series": "EQ"},
    "INE003A01024": {"name": "Siemens Ltd.", "industry": "Capital Goods", "symbol": "SIEMENS", "series": "EQ"},
    "INE343H01029": {"name": "Solar Industries India Ltd.", "industry": "Chemicals", "symbol": "SOLARINDS", "series": "EQ"},
    "INE073K01018": {"name": "Sona BLW Precision Forgings Ltd.", "industry": "Automobile and Auto Components", "symbol": "SONACOMS", "series": "EQ"},
    "INE062A01020": {"name": "State Bank of India", "industry": "Financial Services", "symbol": "SBIN", "series": "EQ"},
    "INE114A01011": {"name": "Steel Authority of India Ltd.", "industry": "Metals & Mining", "symbol": "SAIL", "series": "EQ"},
    "INE044A01036": {"name": "Sun Pharmaceutical Industries Ltd.", "industry": "Healthcare", "symbol": "SUNPHARMA", "series": "EQ"},
    "INE660A01013": {"name": "Sundaram Finance Ltd.", "industry": "Financial Services", "symbol": "SUNDARMFIN", "series": "EQ"},
    "INE195A01028": {"name": "Supreme Industries Ltd.", "industry": "Capital Goods", "symbol": "SUPREMEIND", "series": "EQ"},
    "INE040H01021": {"name": "Suzlon Energy Ltd.", "industry": "Capital Goods", "symbol": "SUZLON", "series": "EQ"},
    "INE494B01023": {"name": "TVS Motor Company Ltd.", "industry": "Automobile and Auto Components", "symbol": "TVSMOTOR", "series": "EQ"},
    "INE092A01019": {"name": "Tata Chemicals Ltd.", "industry": "Chemicals", "symbol": "TATACHEM", "series": "EQ"},
    "INE151A01013": {"name": "Tata Communications Ltd.", "industry": "Telecommunication", "symbol": "TATACOMM", "series": "EQ"},
    "INE467B01029": {"name": "Tata Consultancy Services Ltd.", "industry": "Information Technology", "symbol": "TCS", "series": "EQ"},
    "INE192A01025": {"name": "Tata Consumer Products Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "TATACONSUM", "series": "EQ"},
    "INE670A01012": {"name": "Tata Elxsi Ltd.", "industry": "Information Technology", "symbol": "TATAELXSI", "series": "EQ"},
    "INE155A01022": {"name": "Tata Motors Ltd.", "industry": "Automobile and Auto Components", "symbol": "TATAMOTORS", "series": "EQ"},
    "INE245A01021": {"name": "Tata Power Co. Ltd.", "industry": "Power", "symbol": "TATAPOWER", "series": "EQ"},
    "INE081A01020": {"name": "Tata Steel Ltd.", "industry": "Metals & Mining", "symbol": "TATASTEEL", "series": "EQ"},
    "INE142M01025": {"name": "Tata Technologies Ltd.", "industry": "Information Technology", "symbol": "TATATECH", "series": "EQ"},
    "INE669C01036": {"name": "Tech Mahindra Ltd.", "industry": "Information Technology", "symbol": "TECHM", "series": "EQ"},
    "INE280A01028": {"name": "Titan Company Ltd.", "industry": "Consumer Durables", "symbol": "TITAN", "series": "EQ"},
    "INE685A01028": {"name": "Torrent Pharmaceuticals Ltd.", "industry": "Healthcare", "symbol": "TORNTPHARM", "series": "EQ"},
    "INE813H01021": {"name": "Torrent Power Ltd.", "industry": "Power", "symbol": "TORNTPOWER", "series": "EQ"},
    "INE849A01020": {"name": "Trent Ltd.", "industry": "Consumer Services", "symbol": "TRENT", "series": "EQ"},
    "INE974X01010": {"name": "Tube Investments of India Ltd.", "industry": "Automobile and Auto Components", "symbol": "TIINDIA", "series": "EQ"},
    "INE628A01036": {"name": "UPL Ltd.", "industry": "Chemicals", "symbol": "UPL", "series": "EQ"},
    "INE481G01011": {"name": "UltraTech Cement Ltd.", "industry": "Construction Materials", "symbol": "ULTRACEMCO", "series": "EQ"},
    "INE692A01016": {"name": "Union Bank of India", "industry": "Financial Services", "symbol": "UNIONBANK", "series": "EQ"},
    "INE854D01024": {"name": "United Spirits Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "UNITDSPR", "series": "EQ"},
    "INE200M01039": {"name": "Varun Beverages Ltd.", "industry": "Fast Moving Consumer Goods", "symbol": "VBL", "series": "EQ"},
    "INE205A01025": {"name": "Vedanta Ltd.", "industry": "Metals & Mining", "symbol": "VEDL", "series": "EQ"},
    "INE669E01016": {"name": "Vodafone Idea Ltd.", "industry": "Telecommunication", "symbol": "IDEA", "series": "EQ"},
    "INE226A01021": {"name": "Voltas Ltd.", "industry": "Consumer Durables", "symbol": "VOLTAS", "series": "EQ"},
    "INE075A01022": {"name": "Wipro Ltd.", "industry": "Information Technology", "symbol": "WIPRO", "series": "EQ"},
    "INE528G01035": {"name": "Yes Bank Ltd.", "industry": "Financial Services", "symbol": "YESBANK", "series": "EQ"},
    "INE758T01015": {"name": "Zomato Ltd.", "industry": "Consumer Services", "symbol": "ZOMATO", "series": "EQ"},
    "INE010B01027": {"name": "Zydus Lifesciences Ltd.", "industry": "Healthcare", "symbol": "ZYDUSLIFE", "series": "EQ"}
}
# Data Configuration
INTERVALS = {
    "short_term": "1D",    # Daily for short-term analysis (3-6 months)
    "long_term": "1W"      # Weekly for long-term analysis (>1 year)
}
SHORT_TERM_LOOKBACK = 180  # ~6 months in trading days
LONG_TERM_LOOKBACK = 365   # 1 year

# Technical Indicator Parameters
INDICATORS = {
    # Trend Indicators
    "moving_averages": {
        "ema_short": 9,
        "ema_long": 21,
        "sma_mid": 50,
        "sma_long": 200
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "supertrend": {
        "period": 10,
        "multiplier": 3
    },
    "parabolic_sar": {
        "acceleration_factor": 0.02,
        "max_acceleration_factor": 0.2
    },
    "aroon": {
        "period": 25,
        "uptrend_threshold": 70,
        "downtrend_threshold": 30
    },
    
    # Momentum Indicators
    "rsi": {
        "period": 14,
        "oversold": 30,
        "overbought": 70
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "oversold": 20,
        "overbought": 80
    },
    "roc": {
        "period": 10
    },
    
    # Volatility Indicators
    "bollinger_bands": {
        "period": 20,
        "std_dev": 2
    },
    "atr": {
        "period": 14,
        "multiplier": 1.5
    }
}

# Signal Thresholds
SIGNAL_STRENGTH = {
    "weak": 1,      # One indicator showing buy/sell
    "moderate": 2,  # Two indicators showing buy/sell
    "strong": 3,    # Three or more indicators showing buy/sell
    "very_strong": 5  # Five or more indicators showing buy/sell
}

# Minimum signal strength to generate alert
MINIMUM_SIGNAL_STRENGTH = 3

# Candlestick Pattern Recognition Settings
# True = detect this pattern
CANDLESTICK_PATTERNS = {
    "bullish_engulfing": True,
    "bearish_engulfing": True,
    "doji": True,
    "hammer": True,
    "shooting_star": True,
    "morning_star": True,
    "evening_star": True,
}

# Chart Pattern Recognition Settings
CHART_PATTERNS = {
    "head_and_shoulders": True,
    "inverse_head_and_shoulders": True,
    "double_top": True,
    "double_bottom": True,
    "cup_and_handle": True
}

# How often to run the analysis (in hours)
ANALYSIS_FREQUENCY = 1  # Run every hour

# Message Template with MarkdownV2 Safe Formatting
SIGNAL_MESSAGE_TEMPLATE = """
*TRADING SIGNAL ALERT* 

*Stock:* {stock_name} ({stock_symbol})
*Current Price:* ₹{current_price}
*Signal Type:* {signal_type}
*Timeframe:* {timeframe}
*Strength:* {strength}\\/5  (Escaped `/`)

*Technical Indicators:*
{indicators}

*Patterns Detected:*
{patterns}

*Recommendation:*
{recommendation}

*Generated:* {timestamp}
"""
