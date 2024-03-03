import os
import numpy as np
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.getcwd()
hour_csv_path = os.path.join(current_dir,'hour.csv')
df = pd.read_csv(hour_csv_path)

st.title('Bike Sharing Analysis')

def compare_user_on_holiday(df):
    df_user_on_holiday = df.groupby(by="holiday").agg({
    "casual": "mean",
    "registered": "mean"
    })
    return df_user_on_holiday

def mean_rentals_per_month(df, start_month, end_month):
    df_2012 = df[df['yr'] == 1]
    filtered_month_df = df_2012[(df_2012['mnth'] >= start_month) & (df_2012['mnth'] <= end_month)]
    df_average_rentals_per_month_2012 = filtered_month_df.groupby(by="mnth").cnt.mean()
    return df_average_rentals_per_month_2012

def mean_hourly_rentals_by_season(df):
    df_average_hourly_rental_by_season = df.groupby(by="season").cnt.mean().sort_values(ascending=False)
    return df_average_hourly_rental_by_season

#forQuestionOne
with st.container():
    st.subheader('Average Users of Bike Rentals on Holiday and Non-holiday')
    user_on_holiday_df = compare_user_on_holiday(df)

    with st.container():
        st.sidebar.text("Choose holiday or not: ")
        selected_holiday = st.sidebar.selectbox("Select Holiday", ["Non-Holiday", "Holiday"])
        if selected_holiday == "Non-Holiday":
            filtered_holiday_df = user_on_holiday_df.loc[0] 
        else:
            filtered_holiday_df = user_on_holiday_df.loc[1] 
    
    col1, col2 = st.columns(2)
    with col1:
        average_casual_user = filtered_holiday_df.casual.mean()
        st.metric("Average Casual", value=f"{average_casual_user:.5f}")

    with col2:
        average_registered_user = filtered_holiday_df.registered.mean()
        st.metric("Average Registered", value=f"{average_registered_user:.5f}")

    plt.figure(figsize=(10, 6))

    casual_means = user_on_holiday_df['casual']
    registered_means = user_on_holiday_df['registered']
    labels = ['Non-Holiday', 'Holiday']
    x = range(len(labels))

    plt.bar(x, casual_means, width=0.6, color="#00b4d8" ,label='Casual', align='center')
    plt.bar(x, registered_means, width=0.3, color="#f4a261", label='Registered', align='edge')
    plt.xticks(x, labels)
    plt.ylabel('Average Users')
    plt.title('Average Casual and Registered Users on Holiday and Non-holiday')
    plt.legend()

    st.pyplot(plt)


#forQuestionTwo
with st.container():
    st.subheader('Average Monthly Bike Rentals')

    st.sidebar.text("Select month range: ")
    start_month = st.sidebar.slider("Start Month", min_value=1, max_value=12, value=1)
    end_month = st.sidebar.slider("End Month", min_value=1, max_value=12, value=12)
    filtered_month_df = mean_rentals_per_month(df, start_month, end_month)

    average_rental_month = filtered_month_df.mean()
    st.metric("Average Rental", value=f"{average_rental_month:.5f}")

    months = np.arange(start_month, end_month + 1)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_month_df.index, filtered_month_df.values, marker='o', linestyle='-', color="#00b4d8")
    plt.title('Average Monthly Bike Rentals (2012)')
    plt.xlabel('Month')
    plt.ylabel('Average Bike Rentals')
    plt.xticks(months, month_names[start_month-1:end_month])
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)


#forQuestionThree
with st.container():
    st.subheader('Average Bike Rentals by Season')

    hourly_rentals_by_season_df = mean_hourly_rentals_by_season(df)
    st.sidebar.text("Choose which season you want: ")
    selected_season = st.sidebar.selectbox("Select Season", ['springer', 'summer', 'fall', 'winter'])

    season_mapping = {'springer': 1, 
                      'summer': 2, 
                      'fall': 3, 
                      'winter': 4}
    selected_index = season_mapping[selected_season]
    filtered_season_df = hourly_rentals_by_season_df.loc[selected_index]

    average_hour_by_season = filtered_season_df.mean()
    st.metric("Average Rental", value=f"{average_hour_by_season:.5f}")

    plt.figure(figsize=(12, 6))

    colors = ["#00b4d8", "#00b4d8", "#00b4d8", "#00b4d8"]

    season_mapping_reverse = {v: k for k, v in season_mapping.items()}
    season_labels = hourly_rentals_by_season_df.index.map(season_mapping_reverse)
    num = sns.barplot(x=hourly_rentals_by_season_df.values, y=season_labels, hue=season_labels, palette=colors, legend=False)

    for index, value in enumerate(hourly_rentals_by_season_df.values):
        num.text(value, index, f'{value:.3f}', color='black', ha="left")

    plt.ylabel("Season")
    plt.xlabel("Average Rentals")
    plt.title("Average Bike Rentals By Season", loc="center", fontsize=15)
    plt.tick_params(axis='y', labelsize=12)

    st.pyplot(plt)