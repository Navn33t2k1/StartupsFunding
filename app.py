import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from thefuzz import fuzz
import streamlit as st
import plotly.express as px

df = pd.read_csv('Startups_cleaned_df.csv')
df['Startup_lower']= df['StartUp'].str.lower()

investor_df1 = pd.read_csv('investor_cluster.csv')
investor_df1['Avg_Year'] = round(investor_df1['Avg_Year'])
startup_df1 = pd.read_csv('startup_cluster.csv')
startup_df1.rename(columns={'correct_name': 'startup_name'}, inplace=True)
startup_df1['Avg_Year'] = round(startup_df1['Avg_Year'])

unique_startups = []
for startup in df['StartUp']:
  if not any(fuzz.partial_ratio(startup, name)>=80 for name in unique_startups):
    unique_startups.append(startup)

temp_investor = pd.DataFrame(
    {'Investor':[i.strip() for i in df['Investor'].str.split(',').sum() if i.strip()]}
)

unique_investors = []
for investor in temp_investor['Investor']:
  if not any(fuzz.partial_ratio(investor, name)>=80 for name in unique_investors):
    unique_investors.append(investor)



st.set_page_config(layout='wide', page_title='Indian_StartUp_Analysis')

def Startup_Analysis(startup):
    st.title('StartUp Analysis')
    col1, col2, col3, col4, col5, col6, col7= st.columns(7)
    with col1:
        st.write('StartUp Name')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{startup}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        st.write('Industry')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{df[df['StartUp'].str.contains(startup, case=False)]['Vertical'].values[0]}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col5:
        st.write('Num_Deals')
        num_deals = startup_df1[startup_df1['startup_name'].str.contains(startup, case=False)]['Num_Deals'].values[0]
        html = f"<p style='font-size: 18px; font-weight: bold;'>{num_deals}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col3:
        st.write('Location')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{df[df['StartUp'].str.contains(startup, case=False)]['City'].values[0]}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col4:
        st.write('Funding Rounds')
        unique_rounds = []
        for i in df[df['StartUp'].str.contains(startup, case=False)]['Round']:
            if not any(fuzz.ratio(i, j) >= 80 for j in unique_rounds):
                unique_rounds.append(i)
        html = f"<p style='font-size: 18px; font-weight: bold;'>{', '.join(unique_rounds)}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col6:
        st.write('Total Investment in Cr')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{df[df['StartUp'].str.contains(startup, case=False)]['Amount in Cr'].sum()}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col7:
        st.write('Unique Investors')
        unique_invest = startup_df1[startup_df1['startup_name'].str.contains(startup, case=False)]['Unique_Investors'].values[0]
        html = f"<p style='font-size: 18px; font-weight: bold;'>{unique_invest}</p>"
        st.markdown(html, unsafe_allow_html=True)

    st.write('Investor Names')
    unique_investors = []
    for i in df[df['StartUp'].str.contains(startup, case=False)]['Investor']:
        if not any(fuzz.partial_ratio(i, j) >= 60 for j in unique_investors):
            unique_investors.append(i)
    html = f"<p style='font-size: 18px; font-weight: bold;'>{', '.join(unique_investors) + ' ....'}</p>"
    st.markdown(html, unsafe_allow_html=True)


    temp_cluster = startup_df1[startup_df1['startup_name'].str.contains(startup, case=False)]['Cluster'].values[0]
    similar_startups = startup_df1[startup_df1['Cluster'] == temp_cluster].sort_values(by='Total_Investment', ascending=False).loc[:,['startup_name',	'Avg_Investment',	'Total_Investment',	'Num_Deals',	'Unique_Investors']].reset_index(drop=True)[:5]
    st.subheader('Similer StartUps')
    st.dataframe(similar_startups)

def Overall_Analysis():
    st.title('Overall Analysis')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total StartUps', len(unique_startups))
    with col2:
        st.metric('Total Funding', str(round(df['Amount in Cr'].sum()))+'Cr')
    with col3:
        st.metric('Max Funding', str(round(df['Amount in Cr'].max()))+'Cr')
    with col4:
        st.metric('Avg Funding', str(round(df.groupby('Startup_lower')['Amount in Cr'].sum().mean()))+'Cr')

    st.subheader('Monthly Startups and Investment Trends (Year-by-Year)')
    select_option = st.selectbox('Select Type', ['Annual Funding Trends by Month', 'Monthly Funded Startup Counts by Year'])
    if select_option=='Annual Funding Trends by Month':
        temp_df = df.groupby(['Year', 'Month'])['Amount in Cr'].sum().reset_index()
        fig = px.line(temp_df, x='Month', y='Amount in Cr', color='Year', markers=True, range_x=[1,12])
        fig.update_traces(line=dict(width=4), marker=dict(size=10))
        st.plotly_chart(fig)
    else:
        temp_df = df.groupby(['Year', 'Month'])['StartUp'].count().reset_index()
        fig = px.line(temp_df, x='Month', y='StartUp', color='Year', markers=True, range_x=[1, 12])
        fig.update_traces(line=dict(width=4), marker=dict(size=10))
        st.plotly_chart(fig)

    st.subheader('Annual vs. Overall Leading Startups by Funding')
    select_option0 = st.selectbox('Select One', ['Annual Leading Startup by Investment Amount', 'Top 10 Startups by Total Funding Raised'])
    if select_option0=='Annual Leading Startup by Investment Amount':
        year_wise = df.groupby(['Year', 'StartUp'], as_index=False)['Amount in Cr'].sum().sort_values(
            ['Year', 'Amount in Cr'], ascending=[True, False]).drop_duplicates(subset=['Year'],
                                                                               keep='first').reset_index(drop=True)
        fig4 = px.bar(year_wise, x='Year', y='Amount in Cr',color='StartUp')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        top_start = df.groupby('StartUp', as_index=False)['Amount in Cr'].sum().sort_values(by='Amount in Cr', ascending=False)[
        :10].reset_index(drop=True)
        fig = px.bar(top_start, x='StartUp', y='Amount in Cr', color='StartUp', text_auto=True)
        st.plotly_chart(fig)

    st.subheader('Top 5 Investors & City Funding Distribution')
    investor_option = st.selectbox('Select One', ['Top 5 Leading Investors by Total Funding', 'Total Funding Raised Across Top 5 Cities'])
    if investor_option == 'Top 5 Leading Investors by Total Funding':
        top_investors = pd.Series(df['Investor'].str.split(',').apply(lambda x: [i.strip() for i in x if i.strip()]).sum(),
                                  name='Investor')
        investor_dict = {}
        for investor in top_investors:
            mask = df['Investor'].str.contains(investor, case=False, regex=False)
            investor_dict[investor] = df.loc[mask, 'Amount in Cr'].sum()

        investor_df = pd.DataFrame.from_dict(investor_dict, orient='index',
                                             columns=['Total Amount Invested']).reset_index().sort_values(
            by='Total Amount Invested', ascending=False)
        investor_df['index'] = investor_df['index'].str.title()
        investor_df = investor_df.drop_duplicates(subset=['index'], keep='first').rename(
            columns={'index': 'Investors'}).reset_index(drop=True)
        investor_df.iloc[0, 0] = 'Softbank Group'
        investor_df.iloc[3, 0] = 'Sequoia Capital'
        investor_df.iloc[8, 0] = 'Accel Partners'
        investor_df.iloc[12, 0] = 'Accel Partners'
        investor_df.iloc[5, 0] = 'Tiger Global Management'
        investor_df.iloc[31, 0] = 'Sequoia Capital'
        investor_df.iloc[18, 0] = 'Others'
        investor_df.iloc[7, 0] = 'Tencent Holdings'
        investor_df.iloc[47, 0] = 'Sequoia Capital'
        investor_df.iloc[62, 0] = 'Accel Partners'
        investor_df.iloc[1, 0] = 'IDG'
        top_5_investors = investor_df.groupby('Investors', as_index=False)['Total Amount Invested'].sum().sort_values(by='Total Amount Invested', ascending=False)[:5]
        fig5 = px.pie(top_5_investors, names='Investors', values='Total Amount Invested', hole=0.5)
        st.plotly_chart(fig5)
    else:
        top_funding_across_cities = df.groupby('City')['Amount in Cr'].sum().sort_values(ascending=False)[:5].reset_index()
        fig3 = px.pie(top_funding_across_cities, names='City', values='Amount in Cr', hole=0.5)
        st.plotly_chart(fig3)

    st.subheader("Leading Sectors in Investment: Annual Trends & Overall Rankings")
    sector_option = st.selectbox('Select One', ['Leading Sector in Annual Funding Trends', 'Leading Sectors in Overall Investment'])
    if sector_option == 'Leading Sector in Annual Funding Trends':
        tem_df = df.groupby(['Year', 'Vertical'], as_index=False)['Amount in Cr'].sum().sort_values(
            by=['Year', 'Amount in Cr'], ascending=[True, False]).drop_duplicates(subset=['Year'], keep='first')
        fig6 = px.bar(tem_df, x='Year', y='Amount in Cr', color='Vertical')
        st.plotly_chart(fig6)
    else:
        sector_df = df.groupby('Vertical')['Amount in Cr'].sum().sort_values(ascending=False)[:10].reset_index()
        fig7 = px.bar(sector_df, x='Vertical', y='Amount in Cr', color='Vertical')
        st.plotly_chart(fig7)

    st.subheader("Top 5 Funded Startups & Leading Investment Sectors by Year")
    top5 = st.selectbox('Select One', ['Top 5 Funded Startups by Year', 'Top 5 Funded Sectors by Year'])
    if top5=='Top 5 Funded Startups by Year':
        top5_startups = df.groupby(['Year', 'StartUp'], as_index=False)['Amount in Cr'].sum().sort_values(
            ['Year', 'Amount in Cr'], ascending=[True, False]).groupby('Year').head().reset_index(drop=True)
        fig8 = px.sunburst(top5_startups, path=['Year', 'StartUp'], values='Amount in Cr', color='Amount in Cr')
        st.plotly_chart(fig8)
    else:
        top5_sectors = df.groupby(['Year', 'Vertical'], as_index=False)['Amount in Cr'].sum().sort_values(
            by=['Year', 'Amount in Cr'], ascending=[True, False]).groupby('Year').head().reset_index(drop=True)
        fig9 = px.sunburst(top5_sectors, path=['Year', 'Vertical'], values='Amount in Cr', color='Amount in Cr')
        st.plotly_chart(fig9)

    st.subheader("Top 10 Cities with the Highest Number of Startups")
    top_10_cities = df.groupby('City')['StartUp'].count().sort_values(ascending=False)[:10].reset_index()
    fig2 = px.bar(top_10_cities, x='City', y='StartUp', text='StartUp', color='City')
    st.plotly_chart(fig2)

    st.subheader("Monthly Funding Trend Heatmap by Year")
    pivot_table = pd.pivot_table(
        data=df,
        values='Amount in Cr',
        index='Year',
        columns='Month',
        aggfunc='sum'
    )

    fig3 = px.imshow(pivot_table, text_auto=True, labels=dict(color='Total Funding'))
    st.plotly_chart(fig3)

    st.subheader("Normalized List of Startup Funding Rounds")
    temp_df = pd.Series(df['Round'].str.split(',').apply(lambda x: [i.strip() for i in x if i.strip()]).sum(),
                        name='Round')
    unique_rounds = []
    for i in temp_df:
        if not any(fuzz.ratio(i, name) >= 90 for name in unique_rounds):
            unique_rounds.append(i)
    html_str = "<ul>\n"
    for start in unique_rounds:
        html_str += f"<li>{start}</li>\n"
    html_str += "</ul>"

    st.markdown(html_str, unsafe_allow_html=True)

def investor_details(name):
    st.title('Investor Analysis')
    col11, col12, col13, col14, col15, col16= st.columns(6)
    with col11:
        st.write('Investor Name')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{name}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col12:
        st.write('Generally Invests In')
        invest_in = ' and '.join(df[df['Investor'].str.contains(name, case=False)]['Vertical'].value_counts().index[:2])
        html = f"<p style='font-size: 18px; font-weight: bold;'>{invest_in}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col13:
        st.write('Unique Startups')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{investor_df1[investor_df1['Investor'].str.contains(name, case=False)]['Unique_Startups'].values[0]}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col14:
        st.write('Num Deals')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{investor_df1[investor_df1['Investor'].str.contains(name, case=False)]['Num_Deals'].values[0]}</p>"
        st.markdown(html, unsafe_allow_html=True)
    with col15:
        st.write('Total Investment in Cr')
        html = f"<p style='font-size: 18px; font-weight: bold;'>{round(investor_df1[investor_df1['Investor'].str.contains(name, case=False)]['Total_Investment'].values[0], 2)}</p>"
        st.markdown(html, unsafe_allow_html=True)

    with col16:
        st.write('Biggest Investment')
        st.dataframe(df[df['Investor'].str.contains(name, case=False)][['StartUp', 'Amount in Cr']].sort_values(by='Amount in Cr', ascending=False).reset_index(drop=True)[:1])

    st.write('Most Recent Investment')
    st.dataframe(df[df['Investor'].str.contains(name, case=False)][['Date', 'StartUp', 'Vertical', 'City', 'Round', 'Amount in Cr']].reset_index(drop=True).head())

    st.subheader('Similar Investors')
    st.dataframe(investor_df1[investor_df1['Investor'].str.contains(name, case=False)].sort_values('Avg_Investment', ascending=False).iloc[:5, :5])

    if invest_option== 'Generally Invests in Sectors':
        st.subheader('Generally Invests in Sectors')
        sector = df[df['Investor'].str.contains(name, case=False)].groupby('Vertical', as_index=False)[
            'Amount in Cr'].sum().sort_values(by='Amount in Cr', ascending=False)
        sector.rename(columns={'Vertical': 'Sector'}, inplace=True)
        fig10 = px.pie(sector, names='Sector', values='Amount in Cr', hole=0.5)
        st.plotly_chart(fig10)
    elif invest_option== 'Generally Invests in Cities':
        st.subheader('Generally Invests in Cities')
        city = df[df['Investor'].str.contains(name, case=False)].groupby('City')[
            'Amount in Cr'].sum().reset_index().sort_values(by='Amount in Cr', ascending=False)
        fig11 = px.pie(city, names='City', values='Amount in Cr', hole=0.5)
        st.plotly_chart(fig11)
    else:
        st.subheader('Generally Invests in Stages')
        stage = df[df['Investor'].str.contains(name, case=False)].groupby('Round')[
            'Amount in Cr'].sum().reset_index().rename(columns={'Round': 'Stage'}).sort_values(by='Amount in Cr',
                                                                                               ascending=False)
        fig12 = px.pie(stage, names='Stage', values='Amount in Cr', hole=0.5)
        st.plotly_chart(fig12)

    st.subheader('YoY Investment')
    YoY_df = df[df['Investor'].str.contains(name, case=False)].groupby('Year')['Amount in Cr'].sum().reset_index()
    fig14 = px.bar(YoY_df, x='Year', y='Amount in Cr', color='Year')
    fig14.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig14)

st.sidebar.subheader("Indian StartUps Funding Ecosystem")
option = st.sidebar.selectbox('Select One', ['Overall Analysis', 'StartUp Analysis', 'Investor Analysis'])
if option=='Overall Analysis':
    Overall_Analysis()
    st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><hr>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Created by <b>Navneet</b></p>", unsafe_allow_html=True)
elif option=='StartUp Analysis':
    selected_startup = st.sidebar.selectbox('Select StartUp', sorted(unique_startups))
    btn1 = st.sidebar.button('Find StartUp Details')
    if btn1:
        Startup_Analysis(selected_startup)
    st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><hr>",
                        unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Created by <b>Navneet</b></p>", unsafe_allow_html=True)
else:
    selected_investor = st.sidebar.selectbox('Select Investor', sorted(unique_investors))
    invest_option = st.sidebar.selectbox('Select Investor Preferences', ['Generally Invests in Sectors', 'Generally Invests in Cities',
                                                'Generally Invests in Stages'])
    btn2 = st.sidebar.button('Find Investor Details')
    if btn2:
        investor_details(selected_investor)
    st.sidebar.markdown("<br><br><br><br><br><br><hr>",
                        unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Created by <b>Navneet</b></p>", unsafe_allow_html=True)
