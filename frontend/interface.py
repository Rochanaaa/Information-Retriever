import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import date, datetime
from elasticsearch import Elasticsearch
import time
import nltk
from nltk.corpus import stopwords 
from wordcloud import WordCloud

nltk.download('stopwords')     


# To run locally, use command streamlit run interface.py
class App:
  def __init__ (self):
    self.es = Elasticsearch("http://elasticsearch:9200")  # Adjust the connection string as necessary

    self.query = ""
    
    self.no_of_posts = None
    self.min_score = None
    self.max_score = None
    self.upvote_ratio = None
    self.min_date = None
    self.max_date = None
    self.subreddits = None    
    self.include_search = None    
    self.get_all_results = None    

    self.settings_error = False

  def render(self):
    st.set_page_config(
      page_title="Generative AI Opinion Search Engine", 
      layout="wide", 
      menu_items={
        "About": "### This is an opinion search engine built for CZ4034 Information Retrieval."
      },
    )

    self.render_sidebar()
    self.render_query_section()

  def render_query_section(self):  
    st.markdown(f"<h1 style='text-align: center; margin-bottom: 10pt'>Generative AI Opinion Search Engine</h1>", unsafe_allow_html=True)

    query_col1, query_col2 = st.columns([8,1])
    with query_col1:
      query = st.text_input(label="Search", label_visibility='collapsed', value="", placeholder="Enter your query here")
    with query_col2:
      btn = st.button("Search", disabled=self.settings_error, type="primary", use_container_width=True)
    
    if(btn or query):
      self.query = query
      if(query):
        start = time.time()
        with st.spinner("Loading..."):
          posts = self.get_results(
            query, 
            self.no_of_posts, 
            self.min_score, 
            self.max_score, 
            self.upvote_ratio, 
            self.min_date, 
            self.max_date, 
            self.subreddits,         
            self.include_search,
            self.get_all_results         
            )
        end = time.time() 
        self.render_results(posts, end -  start)

  def render_results(self, posts, time_elapsed):
    if(posts.size > 0):
      st.success(f"{posts.shape[0]} results found in {round(time_elapsed, 4)} seconds")

      tab1, tab2, tab3, tab4, tab5 = st.tabs(["All Results", "Objective Results", "Subjective Results", "Positive Results", "Negative Results"])

      # All results
      with tab1:
        self.render_tab(posts.copy(), "all")        

      # Objective results
      with tab2:
        self.render_tab(posts.copy(), "Objective")

      # Subjective results
      with tab3:
        self.render_tab(posts.copy(), "Subjective")        

      # Positive results
      with tab4:
        self.render_tab(posts, "Positive")        
        
      # Negative results
      with tab5:
        self.render_tab(posts, "Negative")        

    else:
      st.error(f"No results found")

  def render_tab(self, data, key):
    _, res_col1, _, res_col2, _ = st.columns([0.1, 3, 0.5, 6, 0.1])
    with res_col1:
      if key == 'all':
        field = 'combined_label'
        labels = data[field].unique()
        colours = self.get_colours(labels, True)
        self.render_pie(data, field, colours=[colours[key] for key in labels], labels=labels)
      elif key == 'Subjective' or key == 'Objective':
        temp = data[data["subjectivity_label"] == "Subjective"].reset_index(drop=True)
        field = 'subjectivity_label'
        self.render_pie(data, field)
        data = data[data[field] == key].reset_index(drop=True)
      else:
        temp = data[data["subjectivity_label"] == "Subjective"].reset_index(drop=True)
        field = 'sentiment_label'
        labels = temp[field].unique()
        colours = self.get_colours(labels, True)
        self.render_pie(temp, field, colours=[colours[key] for key in labels], labels=labels)
        data = data[data[field] == key].reset_index(drop=True)
      st.divider()
      self.render_wordcloud(data)           
      st.divider()
      self.render_pie(data, "subreddit", "Data by subreddit")
    with res_col2:
      self.render_search_results(data, key)       

    # trends by date        
    st.divider()
    self.render_date_chart(data.copy(), f"{key}_all")
    st.divider()
    
    # raw data
    st.subheader("Raw results")
    self.render_raw_data(data, key)

  def render_date_chart(self, data, key):
    data['post_created'] = pd.to_datetime(data['post_created'])
    data = data.dropna(subset=['post_created'])
    
    if data.empty:
      return

    filtered_data = data
    
    min_date = data['post_created'].min().date()
    max_date = data['post_created'].max().date()
    
    st.subheader("Results by date")
    if min_date != max_date:
      dates = st.slider("Date range to display", min_date, max_date, (min_date, max_date), key=f"{key}_slider")
      filtered_data = data[(data['post_created'].dt.tz_localize(None) >= pd.to_datetime(dates[0])) & (data['post_created'].dt.tz_localize(None) <= pd.to_datetime(dates[1]))]
    view = st.selectbox("Group by", ['Date', 'Month', 'Year'], key=f"{key}_select")

    colours = self.get_colours(filtered_data['combined_label'].unique())

    if view == 'Date':
      filtered_data['date'] = filtered_data['post_created'].dt.to_period('D').astype(str)
      filtered_data = filtered_data.groupby(['date', 'combined_label']).size()\
            .unstack(fill_value=0).rename(columns={"date": "index"}) 
    elif view == 'Month':
      filtered_data['month'] = filtered_data['post_created'].dt.to_period('M').astype(str)
      filtered_data = filtered_data.groupby(['month', 'combined_label']).size()\
            .unstack(fill_value=0).rename(columns={"month": "index"}) 
    elif view == 'Year':
      filtered_data['year'] = filtered_data['post_created'].dt.to_period('Y').astype(str)
      filtered_data = filtered_data.groupby(['year', 'combined_label']).size()\
            .unstack(fill_value=0).rename(columns={"year": "index"})  

    tab1, tab2 = st.tabs(["Line", "Histogram"])
    with tab1:
      st.line_chart(filtered_data, height=400, color=colours)
    with tab2:
      st.bar_chart(filtered_data, height=400, color=colours)
      
  @st.cache_data
  def render_wordcloud(_self, df):    
      data = df.copy()
      data['post'] = data['post_title'].fillna('') + " " + data['post_body'].fillna('')
      text = data['post'].astype(str).str.cat(sep=' ')

      if (len(text.split())) == 0:
        st.warning("No words found to generate word cloud")
      else:
        stop_words = list(stopwords.words('english'))
        stop_words.append('nan')
        stop_words.append('-')
        wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=1000, height=500, ).generate(text)
        st.image(wordcloud.to_array(), use_column_width=True)

  @st.cache_data
  def render_pie(_self, data, key, title=None, colours=None, labels=[]):  
    if not colours:
      colours = sb.color_palette("mako")
    pie = data[key].value_counts()
    if len(labels) == 0:
      labels = pie.index
    fig, ax = plt.subplots()
    ax.pie(pie, labels=labels, colors=colours, autopct='%1.0f%%')
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color="white"))
    st.pyplot(fig)

  def render_search_results(self, data, key):
    # Display 5 search results per page
    total_results = data.shape[0]
    num_results_per_page = None
    current_page = None

    start_index = None
    end_index = None

    bottom_menu = st.columns([2, 1, 1])
    with bottom_menu[2]:
      num_results_per_page = st.selectbox("Page Size", options=[5, 10, 25, 50, 100], key=key)
    with bottom_menu[1]:
      num_pages = total_results // num_results_per_page + (1 if total_results % num_results_per_page > 0 else 0)
      current_page = st.number_input("Page", value=1, min_value=1, max_value=num_pages, key=f"{key}_page")
      start_index = (current_page - 1) * num_results_per_page
      end_index = min(start_index + num_results_per_page, total_results)
    with bottom_menu[0]:
      st.markdown(f"Showing **{start_index+1}**-**{end_index}** of **{total_results}** results")
      st.markdown(f"Page **{current_page}** of **{num_pages}**")
    
    colors = self.get_colours(data['combined_label'].unique(),True)
    for index in range(start_index, end_index):
        row = data.iloc[index]
        sentiment_color = colors[row['combined_label']]

        def extract_date(date_string):
          try:
              # Parse the datetime string
              date_time_obj = datetime.fromisoformat(date_string)
              # Format the date part
              formatted_date = date_time_obj.strftime("%Y-%m-%d")
              return formatted_date
          except ValueError:
              # Return the original string if there's an error
              return date_string
        
        with st.container(border=True):
          formatted_date = extract_date(row['post_created'])
          l, r = st.columns([5,1])
          with l:
            if formatted_date:
                st.text(f"r/{row['subreddit']} • Posted {formatted_date}")
            else:
                st.text(f"r/{row['subreddit']} • Posted {row['post_created']}")
          with r:
            st.markdown(f"<p style='color: white; text-align: center; background-color: {sentiment_color}; border-radius: 4pt'>{row['combined_label']}</p>", unsafe_allow_html=True)
              
          st.markdown(f"<a href={row['post_url']} style='color: {sentiment_color}; font-size: 20pt; font-weight: 600; text-decoration: none'>🔗 {row['post_title']}</a>", unsafe_allow_html=True)
          
          f1, f2 = st.columns(2)
          f1.metric("Score", row['post_score'], help="Number of upvotes received by the post")
          f2.metric("Upvote ratio", row['post_upvote_ratio'], help="Ratio of number of upvotes received by the post to the total number of votes")
          
          if type(row['post_body']) == str:
            with st.expander("Post body"):
              st.write(row['post_body'])

  def highlight_df(self, row):
    if row['combined_label'] == "Positive":
      return ["background-color: lightgreen"] * len(row)
    elif row['combined_label'] == "Negative":
      return ["background-color: salmon"] * len(row)
    else:
      return ["background-color: lightgrey"] * len(row)

  def get_colours(self, labels, dict=False):
    colours = []
    colours_dict = {}
    if 'Negative' in labels:
      colours.append('#f54242')
      colours_dict['Negative'] = '#FA8072'
    if 'Objective' in labels:
      colours.append('#a8a8a8')
      colours_dict['Objective'] = '#A8A8A8'
    if 'Positive' in labels:
      colours.append('#36c23d')
      colours_dict['Positive'] = '#61CC47'
    
    if dict:
      return colours_dict

    return colours

  def render_sidebar(self):
    with st.sidebar:
      with st.form("search_settings", border=False):
        st.subheader  ("Search settings")

        # Number of posts
        self.no_of_posts = st.slider("Number of posts", 0, 5000, 1000, 100, key="no_of_posts") 

        # Option to get all results
        self.get_all_results = st.checkbox("Get all results", False, key="get_all_results") 

        # Post score
        col1, col2 = st.columns(2)
        with col1:
          self.min_score = st.number_input("Minimum score", 0, key="min_score")
        with col2:
          max_score_info = "Indicate a value above 0 to filter for scores"
          # Set the default value to None
          self.max_score = st.number_input("Maximum score", 0, key="max_score", help=max_score_info)
        
        if((self.max_score > 0 and self.min_score > 0) and (self.max_score < self.min_score)):
          st.error("Max score cannot be less than min score")
          self.settings_error = True
        else:
          self.settings_error = False

        # Upvote ratio
        self.upvote_ratio = st.slider("Upvote ratio", 0.0, 1.0, (0.0, 1.0), key="upvote_ratio")  

        # Date range
        col3, col4 = st.columns(2)
        with col3:
          self.min_date = st.date_input(label="Date From", value=None, max_value=date.today(), key="min_date")
        with col4:
          self.max_date = st.date_input(label="Date To", value=None, max_value=date.today(), key="max_date")
        if((not self.min_date == None and not self.max_date  == None) and (self.max_date < self.min_date)):
          st.error("Max date cannot be less than min date")
          self.settings_error = True

        # Subreddits
        subreddit_options = ['GenerativeAI','ChatGPT', 'OpenAI', 'GPT3', 'aiArt', 'NewTech', 'Futurology', 'Technology', 'MachineLearning', 'ArtificialInteligence']
        self.subreddits = st.multiselect("Subreddits to include",
                subreddit_options,subreddit_options)
        if(not self.subreddits):
          st.error("Please select at least one subreddit")
          self.settings_error = True

        # Use title and/or body in search
        self.include_search = st.radio("Include",
          ("Title only", "Body only", "Both title and body"),
          index=2,
          help="What to include in search - Reddit post title, body, or both",          
          key="include_search"
        )

        def apply_changes():
          st.toast("Changes applied")

        st.form_submit_button("Apply changes", on_click=apply_changes,)
  
  def render_raw_data(self, data, key):
    total_results = data.shape[0]
    num_results_per_page = 100
    current_page = None

    start_index = None
    end_index = None

    bottom_menu = st.columns([2, 1])
    with bottom_menu[1]:
      num_pages = total_results // num_results_per_page + (1 if total_results % num_results_per_page > 0 else 0)
      current_page = st.number_input("Page", value=1, min_value=1, max_value=num_pages, key=f"{key}_page_raw")
      start_index = (current_page - 1) * num_results_per_page
      end_index = min(start_index + num_results_per_page, total_results)
    with bottom_menu[0]:
      st.markdown(f"Showing **{start_index+1}**-**{end_index}** of **{total_results}** rows | Page **{current_page}** of **{num_pages}**")
    
    formatted_res = data.iloc[start_index:end_index].style.apply(self.highlight_df, axis=1)
    st.dataframe(data=formatted_res, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
      st.download_button(
        "Download full raw data",
        self.convert_df(data),
        f"{self.query}_data_{key}_full.csv",
        "text/csv",
        key=f'download-full-data_{key}'
      )
    with col2:
      st.download_button(
        "Download data on current page",
        self.convert_df(data.iloc[start_index:end_index]),
        f"{self.query}_data_{key}_from_{start_index}.csv",
        "text/csv",
        key=f'download-current-data_{key}'
      )

  def convert_df(self, df):
    return df.to_csv(index=False).encode('utf-8')

  @st.cache_data
  def get_results(_self, query, no_of_posts, min_score, max_score, upvote_ratio, min_date, max_date, subreddits, include_search, get_all_results):
    if(_self.query):      
    
      es = Elasticsearch("http://elasticsearch:9200")  # Ensure this points to your Elasticsearch instance 
  
      # Build the Elasticsearch query filters based on the user inputs 
      query_filters = [] 

      # Range filters for scores and dates
      # Add score filter
      if _self.min_score is not None and _self.min_score > 0:
          query_filters.append({"range": {"post_score": {"gte": _self.min_score}}})
      if _self.max_score is not None and _self.max_score > 0:
          query_filters.append({"range": {"post_score": {"lte": _self.max_score}}})

      # Date range filters 
      if min_date is not None: 
          query_filters.append({"range": {"post_created": {"gte": _self.min_date.isoformat()}}}) 
      if max_date is not None: 
          query_filters.append({"range": {"post_created": {"lte": _self.esmax_date.isoformat()}}}) 

      # Upvote ratio filter 
      if upvote_ratio is not None: 
          min_ratio, max_ratio = _self.upvote_ratio 
          query_filters.append({"range": {"post_upvote_ratio": {"gte": min_ratio, "lte": max_ratio}}}) 

      # Subreddit filter 
      if subreddits: 
          query_filters.append({"terms": {"subreddit": _self.subreddits}})

      # Custom scoring functions 
      score_functions = [] 
      # if newer: 
      score_functions.append({ 
          "gauss": { 
              "post_created": { 
                  "origin": "now",
                    "scale": "30d",
                    "offset": "15d",
                    "decay": 0.5
              } 
          } 
      }) 
      # if higher_score: 
      score_functions.append({ 
          "field_value_factor": { 
              "field": "post_score", 
              "factor": 2, 
              "modifier": "sqrt", 
              "missing": 1 
          } 
      }) 

      # Define how to search (boosting fields)
      include_in_search = [] 
      if include_search == "Title only": 
          include_in_search = [{"match": {"post_title": {"query": query, "boost": 3}}}]
      elif include_search == "Body only": 
          include_in_search = [{"match": {"post_body": {"query": query, "boost": 2}}}]
      else: 
          include_in_search = [ 
              {"match": {"post_title": {"query": query, "boost": 3}}}, 
              {"match": {"post_body": {"query": query, "boost": 2}}} 
          ]
        
      # Build the complete search request with function score 
      search_body = { 
          "query": { 
              "function_score": { 
                  "query": { 
                      "bool": { 
                          # "should": [{"match": {"_all": query}}],
                          "should": include_in_search, 
                          "minimum_should_match": 1, 
                          "filter": query_filters 
                      } 
                  }, 
                  "functions": score_functions, 
                  "score_mode": "multiply", 
                  "boost_mode": "multiply",
                  "boost": 1,
                  "max_boost": 3,  # Adjust as needed
                  "min_score": min_score
              } 
          }
      } 

       # If user does not specify they want to get all results
      if not get_all_results:
        search_body['size'] = no_of_posts 
      else:
        search_body['size'] = 14640 

      # Execute the search 
      response = es.search(index="posts", body=search_body) 

      # Extract the search hits 
      hits = response['hits']['hits'] 
      results = [hit["_source"] for hit in hits] 

      return pd.DataFrame(results)

if(__name__ == "__main__"):
  app = App()
  app.render()
