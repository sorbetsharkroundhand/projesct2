import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 데이터 로딩
file_path = 'ratings.dat' 
column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings = pd.read_csv(file_path, sep='::', header=None, names=column_names, engine='python')

# 행렬 생성
num_users = 6040
num_movies = 3952
rating_matrix = pd.DataFrame(0, index=range(1, num_users + 1), columns=range(1, num_movies + 1))

# 값이 있으면 채우고, 없으면 0인상태로
for row in ratings.itertuples(index=False):
    user_id, movie_id, rating = row[0], row[1], row[2]  
    if user_id <= num_users and movie_id <= num_movies:
        rating_matrix.at[user_id, movie_id] = rating

# kmeans 로 불류
kmeans = KMeans(n_clusters=3, random_state=100)
user_clusters = kmeans.fit_predict(rating_matrix)

# 0, 1, 2로 군집화 된 애들끼리 다시 행렬 생성
cluster_0 = rating_matrix.iloc[np.where(user_clusters == 0)]
cluster_1 = rating_matrix.iloc[np.where(user_clusters == 1)]
cluster_2 = rating_matrix.iloc[np.where(user_clusters == 2)]

# 데이터 준비완료


# 함수

def AU(cluster):
    return cluster.sum().nlargest(10)

def AVG(cluster):
    return cluster.mean().nlargest(10)

def SC(cluster):
    return (cluster > 0).sum().nlargest(10)

def AV(cluster, threshold=4):
    return (cluster >= threshold).sum().nlargest(10)

def BC(cluster):
    n_items = cluster.shape[1]
    ranks = cluster.apply(lambda x: x.rank(method='min', ascending=False), axis=1)
    borda_scores = np.zeros(n_items)

    for i in range(cluster.shape[0]):
        user_ranks = ranks.iloc[i]
        max_rank = user_ranks.max()
        rank_counts = user_ranks.value_counts()

        for rank in range(1, int(max_rank) + 1):
            items_with_rank = (user_ranks == rank).sum()
            if items_with_rank > 1:
                points = (n_items - rank) / items_with_rank
                borda_scores[user_ranks[user_ranks == rank].index - 1] += points
            else:
                borda_scores[user_ranks[user_ranks == rank].index - 1] += (n_items - rank)

    return pd.Series(borda_scores, index=cluster.columns).nlargest(10)

# 만약 너무 오래걸리면 이부분은 주석처리해주세요, 제 컴퓨터에선 1시간 정도 걸렸습니다.
def CR(cluster):
    item_scores = np.zeros(cluster.shape[1])
    for i in range(cluster.shape[1]):
        for j in range(cluster.shape[1]):
            if i != j:
                i_wins = (cluster.iloc[:, i] > cluster.iloc[:, j]).sum()
                j_wins = (cluster.iloc[:, j] > cluster.iloc[:, i]).sum()
                if i_wins > j_wins:
                    item_scores[i] += 1
                elif i_wins < j_wins:
                    item_scores[i] -= 1
    return pd.Series(item_scores, index=cluster.columns).nlargest(10)


# 클러스터별로 진행 3*6행렬 만들기 !
clusters = [cluster_0, cluster_1, cluster_2]
results = {}

for i, cluster in enumerate(clusters):
    au_results = AU(cluster)
    avg_results = AVG(cluster)
    sc_results = SC(cluster)
    av_results = AV(cluster)
    bc_results = BC(cluster)
    cr_results = CR(cluster)
    
    combined_results = pd.DataFrame({

        'AU': au_results.values,

        'Avg': avg_results.values,

        'SC': sc_results.values,

        'AV': av_results.values,

        'BC': bc_results.values,

        'CR': cr_results.values
    }).T
    
    results[f'cluster_{i}'] = combined_results

# Display results
for cluster, df in results.items():
    print(f"Results for {cluster}:\n") # 가시성을 위해 클러스터별 구분표시를 해뒀습니다. 양해바람.
    print(df)
    print("\n")
