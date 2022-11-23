# import module
from asyncio.windows_events import NULL
from operator import index
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

#importing the model
model = joblib.load('./Model.sav')

type=st.sidebar.selectbox('Type of Prediction',('Single Customer','Multiple Customer'))
#read data from original csv file
dupli_data=  pd.read_csv('E Commerce Dataset.csv')
#replace null value
dupli_data=dupli_data.interpolate(method = "linear")
#preprocess the numerical data and categorical data

def pre_processing(df):
    #numerical data
    num=['Tenure','NumberOfDeviceRegistered','NumberOfAddress','Complain','CashbackAmount']
    for i in num:
        row =[]
        for j in dupli_data[i]:
            row.append([j])
        row.append([df[i]])
        tenure = np.array(row)
        sc = MinMaxScaler()
        tenure=sc.fit_transform(tenure)
        df[i]=tenure[len(tenure)-1]

    #categorical data
    df['PreferredLoginDevice']= df['PreferredLoginDevice'].map({'Mobile Phone':1,'Computer':0})
    df['CityTier']= df['CityTier'].map({'3':2,'1':0,'2':1})
    df['PreferredPaymentMode']= df['PreferredPaymentMode'].map({'Debit Card':3,'CC':0,'COD':1,'E wallet':4,'Credit Card':2})
    df['Gender']= df['Gender'].map({'Male':1,'Female':0})
    df['PreferedOrderCat']= df['PreferedOrderCat'].map({'Laptop & Accessory':2,'Mobile Phone':3,'Others':4,'Fashion':0,'Grocery':1})
    df['SatisfactionScore']= df['SatisfactionScore'].map({'2':1,'3':2,'5':4,'4':3,'1':0})
    df['MaritalStatus']= df['MaritalStatus'].map({'Single':2,'Divorced':0,'Married':1})

    return df

if type == 'Single Customer':

    tenure = st.number_input('Number of months customer is relationship with company :',0,100)
    preferred_login_device = st.selectbox('Customer prefered login device : ',('Mobile Phone','Computer'))
    citytier = st.radio('City tier of customer  : ',('1','2','3'))
    warehousetohome = st.number_input('Distance in between warehouse to home of customer',0,1000)
    preferred_payment_method = st.selectbox("Preferred payment by the customer : ",('Debit Card','CC','COD','E wallet','Credit Card'))
    gender = st.radio('Gender  : ',('Male','Female'))
    number_of_device_reg = st.number_input('Number of device registered by the Customer :',0,100)
    preferred_ordered_category = st.selectbox('Prefered order category  by the Customer :',('Laptop & Accessory','Mobile Phone','Others','Fashion','Grocery'))
    satisfaction_score = st.radio('Saisfication Score given by the customer  : ',('1','2','3','4','5'))
    marital_status = st.radio('MaritalStatus : ',('Single','Divorced','Married'))
    number_complains = st.number_input('Number of complains by the Customer on the product :',0,100)
    number_of_address = st.number_input('Number of address registered by the Customer :',0,100)
    order_amount_hike = st.number_input('Order Amount Hike From lastYear (Increases in order from last year) :',0,1000)
    number_of_coupon = st.number_input('Number of coupon used by the Customer :',0,100)
    number_of_orders = st.number_input('Number of orders by the Customer :',0,100)
    day_since_order = st.number_input('Day Since Last Order :',0,1000)
    Cask_back_amount = st.number_input('Cash back amount for the Customer :',0,10000)

    preddata={
    'Tenure': tenure,
    'PreferredLoginDevice': preferred_login_device,
    'CityTier': citytier,
    'WarehouseToHome': warehousetohome,
    'PreferredPaymentMode': preferred_payment_method,
    'Gender': gender,
    'NumberOfDeviceRegistered': number_of_device_reg,
    'PreferedOrderCat': preferred_ordered_category,
    'SatisfactionScore': satisfaction_score,
    'MaritalStatus': marital_status,
    'NumberOfAddress': number_of_address,
    'Complain': number_complains,
    'OrderAmountHikeFromlastYear': order_amount_hike,
    'CouponUsed': number_of_coupon,
    'OrderCount': number_of_orders,
    'DaySinceLastOrder': day_since_order,
    'CashbackAmount': Cask_back_amount
    }

    st.write('Overview of the data that entered : ')
    df = pd.DataFrame.from_dict([preddata])
    st.dataframe(df)

    #Prediction
    if st.button('Predict'):
        if(model.predict(pre_processing(df))[0]==1):
            st.warning('Customer will churn')
        else:
            st.success('customer will not churn')
            st.balloons()

else:
     dataa = st.file_uploader('Upload the file ',type=['.csv'])
     multi_customer_df = pd.DataFrame(columns=['Customer ID','Prediction'])

     if dataa is not None:
        test_data= pd.read_csv(dataa)
        st.dataframe(test_data)
        customer_id = test_data['Customer ID']
        data_set = test_data.drop(['Customer ID'],axis =1)
        for i in range(0,len(test_data)):
            roww= test_data.iloc[i].values
            preddata={
            'Tenure': roww[1],
            'PreferredLoginDevice':roww[2],
            'CityTier': roww[3],
            'WarehouseToHome': roww[4],
            'PreferredPaymentMode': roww[5],
            'Gender': roww[6],
            'NumberOfDeviceRegistered': roww[7],
            'PreferedOrderCat': roww[8],
            'SatisfactionScore': roww[9],
            'MaritalStatus': roww[10],
            'NumberOfAddress': roww[11],
            'Complain': roww[12],
            'OrderAmountHikeFromlastYear': roww[13],
            'CouponUsed': roww[14],
            'OrderCount': roww[15],
            'DaySinceLastOrder': roww[16],
            'CashbackAmount': roww[17]
            }
            df = pd.DataFrame.from_dict([preddata])
            df['CityTier'] = df['CityTier'].astype('str')
            df['SatisfactionScore'] = df['SatisfactionScore'].astype('str')
            pre_data=pre_processing(df)
            pre_diction=model.predict(pre_data)
            if(pre_diction[0]==1):
                churning='Customer will churn'
            else:
               churning='customer will not churn'
            multi_customer_df.loc[len(multi_customer_df)]=(customer_id.values[i],churning)

        if st.button('Predict'):
            st.dataframe(multi_customer_df)