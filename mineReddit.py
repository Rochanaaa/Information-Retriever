# imports
import praw
import pandas as pd
import sqlite3 as sq

# Get credentials from file
f = open("passwords.txt", 'r')
red_client_id = f.readline().strip()
red_client_secret = f.readline().strip()
red_user_agent = f.readline().strip()
f.close()

reddit = praw.Reddit(
    client_id=red_client_id,
    client_secret=red_client_secret,
    user_agent=red_user_agent)

conn = sq.connect('reddit2.db')
cur = conn.cursor()
statement = '''Insert or ignore into Posts values '''

# Check if table already exists
flag = cur.execute("""SELECT tbl_name FROM sqlite_master WHERE type='table' AND tbl_name='Posts'; """).fetchall()
if(flag==[]):
    cur.execute("Create table Posts(Title Text, Body Text, Id Text, Score Integer, Date Integer, URL text, Primary Key(Id))")

subreds = ['careerguidance','jobs','careeradvice','AskHR','TechCareers','FinancialCareers','Consulting']
for i in reddit.subreddit('+'.join(subreds)).new(limit=None):
    cur.execute(statement+str((i.title.replace('"',"'"), i.selftext.replace('"',"'"), i.id, i.score, int(i.created_utc),i.url))+";")
conn.commit()
conn.close()