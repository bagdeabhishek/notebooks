import csv
import datetime
import os
import stat
from collections import Counter
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import psycopg2
import psycopg2.extras
import seaborn as sns
import tweepy
from tqdm.notebook import tqdm
from wordcloud import WordCloud


class TwitterUtil:
    DATABASE = {
        "name": "abhishek",
        "username": "abhishek",
        "password": "vaishu",
        "host": "localhost",
        "port": "5432"
    }
    FONT_PATH = "combinedFont.ttf"
    TWITTER_AUTH_FILE = "/home/abhishek/Documents/TweetCrawlMultiThreaded/twitteraccesscodes.csv"
    CLUSTERS_OF_INTEREST = [36041, 65124]
    MRH_FILE_PATH = "pickles/mention_retweet_hastags(trending).pkl"
    MRH_TIME_FILE_PATH = "pickles/mention_retweet_hastags_timeobj(trending).pkl"
    CLUSTER0_LINK_PATH = "files/cluster36041.link"
    CLUSTER1_LINK_PATH = "files/cluster65124.link"
    CLUSTER0_LINK_IMPORTANCE_PATH = "files/cluster_36041(trending).urls.importance"
    CLUSTER1_LINK_IMPORTANCE_PATH = "files/cluster_65124(trending).urls.importance"
    CLUSTER0_TEXT_IMPORTANCE_PATH = "files/cluster_36041(trending).text.importance"
    CLUSTER1_TEXT_IMPORTANCE_PATH = "files/cluster_65124(trending).text.importance"
    EXCLUDE_LIST = [
        'twitter.com',
        'bit.ly',
        'www.facebook.com',
        'youtu.be',
        'www.instagram.com',
        'www.youtube.com',
        'goo.gl',
        'fb.me',
        'ow.ly',
        'fllwrs.com}\n',
        'fllwrs.com',
        'www.amazon.com',
        'instagram.com',
        'www.pscp.tv'
    ]

    def get_credentials(self):
        """
        Get Twitter Authentication credentials from csv file whose location is given by TWITTER_AUTH_FILE variable
        Returns
        -------
        list of dictionaries containing credentials

        """
        credentials_list = []
        with open(self.TWITTER_AUTH_FILE) as f:
            cr = csv.reader(f)
            keys = next(cr)
            for row in cr:
                dct = {}
                for i in range(len(row)):
                    dct[keys[i]] = row[i]
                credentials_list.append(dct)
        return credentials_list

    def get_tweepy_api(self):
        """
        get a list tweepy api Objects which can be used to run tweepy queries
        Returns
        -------
        list of tweepy api objects

        """
        api_list = []
        for dct in self.get_credentials():
            consumer_key = dct['consumer_key']
            consumer_secret = dct['consumer_secret']
            access_token = dct['access_token']
            access_token_secret = dct['access_token_secret']
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            api_list.append(tweepy.auth.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True))
        return api_list

    def pg_get_conn(self):
        """Get Postgres connection for the database dictionary in the class.
        Returns:
            Connection object : returns Post gres connection object

        """
        try:
            conn = psycopg2.connect(database=self.DATABASE['name'],
                                    user=self.DATABASE['username'], password=self.DATABASE['password'],
                                    host=self.DATABASE['host'], port=self.DATABASE['port'])
            return conn
        except Exception as e:
            print(str(e))

    def run_query(self, query: str = """Select * from tweets_cleaned""", realdict: bool = False, arg: list = None):
        """
        Run a query on the connected database
        :param query: query to execute
        :type query: string
        :param realdict: set True if you want the results as list of dictionary
        :type realdict: bool
        :param arg: Arguments if any needed to be passed to the query
        :type arg: list
        :return: List of tuples consisting of query results
        :rtype: List
        """
        with self.pg_get_conn() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) if realdict else conn.cursor()
            print(query) if not arg else print(cur.mogrify(query, (arg,)))
            cur.execute(query) if not arg else cur.execute(query, (arg,))
            try:
                ans = cur.fetchall()
            except psycopg2.ProgrammingError as e:
                ans = None
            return ans

    def create_graph(self, ls_tup: list) -> nx.DiGraph():
        """
        Create a Directed graph from the list of crawled tweets
        :param ls_tup: list of dictionary or tuples where first element is user and
        second handle is of user whose tweet was retweeted
        :type ls_tup:
        :return:
        :rtype:
        """
        G = nx.DiGraph()
        for dc in tqdm(ls_tup):
            if isinstance(ls_tup, dict):
                tfrom = dc['tweet_from']
                rt = dc['retweeted_status_user_handle']
            else:
                tfrom = dc[0]
                rt = dc[1]
            if G.has_edge(tfrom, rt):
                G[tfrom][rt]['weight'] += 1
            else:
                G.add_edge(tfrom, rt, weight=1)
        return G

    def hashtag_mention_accumulator(self, series, limit=None, delimiter=","):
        """
        This splits the hashtags or mentions separated by comma. Use with apply function
        Parameters
        ----------
        delimiter : the delimiter used to split the string
        series : series passed in apply.
        limit : total number of words to include

        Returns
        -------
        dictionary of hashtags/mentions with frequencies.

        """
        c = Counter()
        for sentence in series:
            if sentence:
                sent_list = sentence.split(delimiter)
                c.update(sent_list)
        result_dict = dict(c.most_common()) if not limit else dict(c.most_common(limit))
        return pd.Series(result_dict)

    def split_list(self, series, handleBool=True):
        handles = []
        listNoOfX = []
        for groupList in series:
            for handle, x in groupList:
                handles.append(handle)
                listNoOfX.append(x)
        if handleBool:
            return handles
        else:
            return listNoOfX

    def get_column_counts_by_cluster(self, df, column_name="retweets", limit=None):
        """
        Get the occurrences of particular string in any column in the dataframe. You can plot these counts in a bar graph using tu.get_barcharts()

        Parameters
        ----------
        df : dataframe read from MRH_TIME_FILE_PATH
        column_name : name of column name to analyze
        limit : limit of records

        Returns
        -------
        dataframe containing counts of each individual value in column and cluster

        """
        wf = df.groupby("cluster")[column_name].apply(self.hashtag_mention_accumulator,
                                                      limit=limit).reset_index().rename(
            columns={'handle': 'noOfX', 'level_1': 'handle'})
        return wf

    def get_barcharts(self, df, column_name="retweets", limit=50, return_val=False):
        """
        Plot the barcharts for tweets from clusters. The barcharts plot the top 50 retweets or mentions from each cluster.
        Parameters
        ----------
        df : the Pandas dataframe containing tweets.
        column_name : Name of the column to plot the barchart. Can be "retweets" or "mentions"
        limit: the no of records these barcharts should be limited to
        return_val: if set to true will also return the dataframe used to plot bar graph, default is false
        Returns
        ---------
        Dataframe containing the results plotted if return_val==True else None

        """
        wf2 = self.get_column_counts_by_cluster(df, column_name, limit)
        clusters = wf2.cluster.unique()
        sns.set(rc={'figure.figsize': (40, 10)})
        i = 0
        f, ax = plt.subplots(len(clusters), 1, figsize=(40, 100))
        f.tight_layout(pad=6.0)
        for cid in clusters:
            g = sns.barplot(x="handle", y="noOfX", hue="cluster", data=wf2[wf2.cluster == cid], ax=ax[i])
            g.set_xticklabels(g.get_xticklabels(), rotation=50, horizontalalignment='right')
            i += 1
        return wf2 if return_val else None

    def plot_word_cloud(self, word_freq_dict, background_color="white", width=800, height=1000, max_words=300,
                        figsize=(50, 50), wc_only=False, color_map="viridis"):
        """
        Plot the word cloud from dictionary
        Parameters
        ----------
        word_freq_dict : dictionary containing words as keys and counts as values.
        background_color :
        width :
        height :
        max_words :
        figsize :
        wc_only :
        color_map :

        Returns
        -------

        """
        word_cloud = WordCloud(background_color=background_color, width=width, height=height,
                               max_words=max_words, colormap=color_map,
                               font_path=self.FONT_PATH).generate_from_frequencies(
            frequencies=word_freq_dict)
        if wc_only:
            return word_cloud
        plt.figure(figsize=figsize)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def reset_everything(self, source_table, min_degree=1, suffix=str(datetime.datetime.now())):
        """
        Reset everything from a new Twitter data table
        Parameters
        ----------
        source_table : New table from which to fetch the data
        min_degree : minimum degree below which all nodes are removed from the network Graph
        suffix : suffix for the newly created files

        Returns
        -------

        """
        ls = self.run_query(
            "Select tweet_from,retweeted_status_user_handle from {} where retweeted_status_user_handle is not null".format(
                source_table))
        G = self.create_graph(ls)
        remove_nodes = [x[0] for x in G.degree(weight='weight') if x[1] <= min_degree]
        G.remove_nodes_from(remove_nodes)
        nx.write_gexf(G, "graphs/G_{0}.gexf".format(suffix))
        self.update_pickle_files(source_table, suffix)
        self.write_info_to_file(source_table, suffix)

    def write_info_to_file(self, tablename, suffix=str(datetime.datetime.now()), text_columns=['text', 'urls']):
        """
        Generate the text files for each cluster of interest.
        Parameters
        ----------
        tablename : Name of the table from which to generate data
        suffix : identifier for the generated text files
        text_columns : columns which to dump in text file

        Returns
        -------

        """
        file_name = "files/cluster_{}({}).{}.importance"
        for column in text_columns:
            cluster_id = 0
            for cluster in self.CLUSTERS_OF_INTEREST:
                try:
                    f = open(file_name.format(cluster, suffix, column), 'w+')
                    absolute_path = os.path.realpath(f.name)
                    f.close()
                    os.chmod(absolute_path, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
                    query = "COPY (SELECT t.{},c.importance from {} t JOIN cluster_mapping c ON t.tweet_from = c.id where c.cluster = {}) TO '{}';".format(
                        column, tablename, cluster, absolute_path)
                    self.run_query(query)
                    if cluster_id % 2 == 0:
                        self.CLUSTER0_LINK_IMPORTANCE_PATH = absolute_path
                    elif cluster_id % 2 == 1:
                        self.CLUSTER1_LINK_IMPORTANCE_PATH = absolute_path
                    cluster_id += 1
                except PermissionError as e:
                    print("Please delete the file {}".format(file_name.format(cluster, suffix, column)))

    def update_pickle_files(self, tablename="tweet_articles_tweepy", suffix=str(datetime.datetime.now())):
        """
        Update the pickle files used for analysis from database table
        Parameters
        ----------
        tablename : Name of the new table
        suffix : suffix for the generated files
        Returns
        -------

        """
        query = "SELECT t.created_at,t.tweet_from,t.user_mentions_name,t.retweeted_status_user_handle,t.hashtags,c.cluster,c.importance FROM {} AS t INNER JOIN cluster_mapping AS c ON t.tweet_from = c.id WHERE c.cluster in %s".format(
            tablename)
        ls = self.run_query(query, arg=tuple(self.CLUSTERS_OF_INTEREST))
        df = pd.DataFrame(ls, columns=['time', 'handle', 'mentions', 'retweets', 'hashtags', 'cluster', 'importance'])
        df['time'] = pd.to_datetime(df['time'], format="%a %b %d %H:%M:%S %z %Y")
        for x in ['mentions', 'hashtags']:
            df[x] = df[x].replace('{}', None)
            df[x] = df[x].str.lstrip('{')
            df[x] = df[x].str.rstrip('}')
        self.MRH_TIME_FILE_PATH = 'pickles/mention_retweet_hastags_timeobj({0}).pkl'.format(suffix)
        df.to_pickle(self.MRH_TIME_FILE_PATH)
        df = df.drop(columns=['time'])
        global MRH_FILE_PATH
        self.MRH_FILE_PATH = 'pickles/mention_retweet_hastags({0}).pkl'.format(suffix)
        df.to_pickle(self.MRH_FILE_PATH)
        return df

    def get_cluster_count(self, series):
        """
        A funtion to get the cluster count in case we group the original dataframe by a certain parameter
        Ex. df.groupby("time").cluster.apply(get_cluster_count)
        Parameters
        ----------
        series : The series passed

        Returns
        -------
        Count for each cluster in the group

        """
        c = Counter()
        c.update(series.to_list())
        return pd.Series(dict(c.most_common()))

    def plot_timeseries_data(self, res, cut_off_date_start=None, cut_off_date_end=None, lineplot=False):
        """
        Plot the data i.e cluster counts after getting result from below operation
        Ex.res =  df.groupby("time").cluster.apply(get_cluster_count)

        Parameters
        ----------
        res : result of this operation df.groupby("time").cluster.apply(get_cluster_count) after resetting index
        cut_off_date_start : the start date for the graph
        cut_off_date_end : the end date for graph
        lineplot : whether to plot a line plot by default set to false which plots scatterplot

        Returns
        -------
        None

        """
        number_of_clusters = len(res.level_1.unique())
        sns.set(rc={'figure.figsize': (40, 30)})
        if lineplot:
            sns.lineplot(x="time", hue="level_1", y="cluster", data=res, legend="full",
                         palette=sns.color_palette("muted", number_of_clusters))
        else:
            sns.scatterplot(x="time", hue="level_1", y="cluster", data=res, legend="full",
                            palette=sns.color_palette("muted", number_of_clusters))
        if cut_off_date_start or cut_off_date_end:
            plt.xlim(cut_off_date_start, cut_off_date_end)

    def get_links_dct(self, file, topk=None, importance=False):
        """
        Get the dictionary representing the count of sources ex. timesofindia.com from a text file
        Parameters
        ----------
        file : The path of the file which has list of urls
        topk : whether you want to return all the sources(None) or limit it top k values
        importance : If set to true increments by importance of particular node instead of treating each node equally and incrementing by 1

        Returns
        -------
        Dictionary with source and its counts

        """
        link_dct = {}
        with open(file) as f:
            for row in f:
                try:
                    if not row.startswith('{}'):
                        url, importance = row.split()
                        importance = float(importance)
                        site_name = url.replace("{", "").replace("}", "").split('/')[2]
                        if importance:
                            if site_name not in link_dct:
                                link_dct[site_name] = importance
                            else:
                                link_dct[site_name] += importance
                        else:
                            if site_name not in link_dct:
                                link_dct[site_name] = 1
                            else:
                                link_dct[site_name] += 1
                except ValueError as e:
                    print(e)
        for x in self.EXCLUDE_LIST:
            link_dct.pop(x, None)
        if topk:
            link_dct = {k: v for k, v in sorted(link_dct.items(), key=lambda item: item[1])[:topk]}
        else:
            link_dct = {k: v for k, v in sorted(link_dct.items(), key=lambda item: item[1])}
        return link_dct

    def plot_wc_subplots(self, cluster0, cluster1, fig_size=(50, 50)):
        """
        Function to plot two wordclouds side by side just to provide easier comparison
        Parameters
        ----------
        cluster0 : dictionary of thing_to_plot as key and its count as value(plotted red)
        cluster1 : dictionary of thing_to_plot as key and its count as value(plotted blue)
        fig_size : tuple denoting size of the figure

        Returns
        -------

        """
        wc0 = self.plot_word_cloud(cluster0, wc_only=True, width=800, color_map="autumn")
        wc1 = self.plot_word_cloud(cluster1, wc_only=True, width=800, color_map="winter")
        fig = plt.figure(figsize=fig_size)
        fig.subplots_adjust(wspace=0)
        s1 = fig.add_subplot(121)
        s1.axis("off")
        s1.imshow(wc0, aspect='auto')
        s2 = fig.add_subplot(122)
        s2.axis("off")
        s2.imshow(wc1, aspect='auto')
        fig.show()

    def get_intersection_balanced_set(self, cluster0, cluster1, cutoff_ratio=0.01):
        """
        remove common elements from dictionaries {key: something, value: count}.
        The values are scaled initially and then we subtract ratio of opposite set from the clusters ratio.
        we remove values from both clusters which fall below the cutoff_ratio
        Parameters
        ----------
        cluster0 : dictionary of counts
        cluster1 : dictionary of counts from the other cluster
        cutoff_ratio : the ratio below which we remove elements from the set. setting it to 0 is basic set intersection

        Returns
        -------
        two dictionaries with rebalanced weights/counts

        """
        common_set = set(cluster0).intersection(set(cluster1))
        total = sum(cluster0.values())
        cluster0 = {k: v / total for k, v in cluster0.items()}
        total = sum(cluster1.values())
        cluster1 = {k: v / total for k, v in cluster1.items()}
        for x in common_set:
            cluster0[x] = round(cluster0[x] - cluster1[x], 3)
            cluster1[x] = round(cluster1[x] - cluster0[x], 3)
            if cluster0[x] <= cutoff_ratio:
                cluster0.pop(x, None)
            if cluster1[x] <= cutoff_ratio:
                cluster1.pop(x, None)
        return cluster0, cluster1

    def plot_wordclouds_side_by_side(self, df, text_column, value_column, cluster_column="cluster"):
        """
        This is utility method to plot wordclouds side by side just by giving the data frame and the key and value columns as parameters.
        This function basically takes care of converting the dataframe into dictionaries.
        Parameters
        ----------
        cluster_column : The column of df which has the cluster number (set to "cluster" by default)
        df : dataframe(Note the cluster column should be present, other wise this will throw exception)
        text_column : the column name of the df which has text that's displayed in WC
        value_column : the name of value clumn which will determine size of words in word cloud.

        Returns
        -------
        None
        """
        value_dict = {}
        for cluster in self.CLUSTERS_OF_INTEREST:
            value_dict[cluster] = dict(
                df[df[cluster_column] == cluster][[text_column, value_column]].to_dict('split')['data'])
        self.plot_wc_subplots(value_dict[self.CLUSTERS_OF_INTEREST[0]], value_dict[self.CLUSTERS_OF_INTEREST[1]])

    def fetch_user_info(self, handle, cluster, api):
        """
        Utility function to get the twitter user information for a single user. For large number of handles use get_twitter_user_info()
        Parameters
        ----------
        handle : the user screen name
        cluster : cluster to which the user belongs to
        api : Tweepy API object

        Returns
        -------
        Pandas series containing all the relevant information

        """
        try:
            user_info = api.get_user(handle)._json
            user_info["cluster"] = cluster
            return pd.Series(user_info)
        except Exception as e:
            return pd.Series(dict({"screen_name": handle, "description": "NOT FOUND", "cluster": cluster}))

    def get_twitter_user_info(self, df_handle, threads=20, handle_column="handle", cluster_column="cluster"):
        """
        Get user information from twitter screen names
        Parameters
        ----------
        df_handle : the DataFrame having handle information (Note: Deduplicate before hand)
        threads : number of threads to use
        handle_column : name of the column containing user screen name
        cluster_column : name of the cluster column

        Returns
        -------
        list of futures objects.

        """
        apis = self.get_tweepy_api()
        results = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for index, row in tqdm(df_handle.iterrows()):
                user_info = executor.submit(self.fetch_user_info, row[handle_column], row[cluster_column],
                                            apis[index % len(apis)])
                results.append(user_info)
        return results

    def __init__(self):
        self.MRH_FILE_PATH = max(glob("pickles/mention_retweet_hastags*.pkl"), key=os.path.getctime)
        self.MRH_TIME_FILE_PATH = max(glob("pickles/mention_retweet_hastags_timeobj*.pkl"), key=os.path.getctime)
        self.CLUSTER0_LINK_PATH = max(glob("files/cluster*.link"), key=os.path.getctime)
        self.CLUSTER1_LINK_PATH = max(glob("files/cluster*.link"), key=os.path.getctime)
        self.CLUSTER0_LINK_IMPORTANCE_PATH = max(glob("files/cluster_*.urls.importance"), key=os.path.getctime)
        self.CLUSTER1_LINK_IMPORTANCE_PATH = max(glob("files/cluster_*.urls.importance"), key=os.path.getctime)
        self.CLUSTER0_TEXT_IMPORTANCE_PATH = max(glob("files/cluster_*.text.importance"), key=os.path.getctime)
        self.CLUSTER1_TEXT_IMPORTANCE_PATH = max(glob("files/cluster_*.text.importance"), key=os.path.getctime)
