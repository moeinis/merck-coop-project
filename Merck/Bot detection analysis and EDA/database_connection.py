# Database connection and data extraction
def fetch_email_engagement_data():
    """
    Fetch email engagement data from the database
    """
    # SQL query to get email engagement metrics
    sql_query = """
    SELECT s.sendid, s.subscriberkey, s.listid, s.batchid, s.emailaddress, 
           s.eventdate as sent_date, s.eventtype, o.eventdate as open_date,
           DATEDIFF(seconds, cast(sent_date as timestamp), cast(open_date as timestamp)) as open_diff,
           o.isunique, c.isuniqueforurl, c.gal_area_code, o.browser, 
           o.emailclient, o.operatingsystem, o.device,
           c.eventdate as click_date, c.eventtype, 
           DATEDIFF(seconds, cast(sent_date as timestamp), cast(click_date as timestamp)) as click_diff,
           ROW_NUMBER() OVER (PARTITION BY s.sendid, s.subscriberkey, s.listid, s.batchid, s.emailaddress 
                             ORDER BY s.emailaddress, open_diff ASC) as rn
    FROM cim_lc_pub.vw_c039_015_sent s
    INNER JOIN cim_lc_pub.vw_c039_011_opens o
        ON s.sendid = o.sendid
        AND s.subscriberkey = o.subscriberkey
        AND s.listid = o.listid
        AND s.batchid = o.batchid
        AND s.emailaddress = o.emailaddress
    INNER JOIN cim_lc_pub.vw_c039_004_clicks c
        ON s.sendid = c.sendid
        AND s.subscriberkey = c.subscriberkey
        AND s.listid = c.listid
        AND s.batchid = c.batchid
        AND s.emailaddress = c.emailaddress
    ORDER BY emailaddress, open_diff ASC;
    """
    
    # Database connection string
    db_string = "postgresql+psycopg2://cim_dwh_ro:PCimDwhRo1234$@awsapcimbirsp01.coaaq18eo1zb.ap-southeast-1.redshift.amazonaws.com:25881/apbirsp01"
    
    # Create connection and fetch data
    db = create_engine(db_string)
    df = pd.read_sql(sql_query, db)
    
    return df

# Fetch the data
df = fetch_email_engagement_data()