
##################################
# Association Rule Based Recommender System
#######################################

# İş Problemi
# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

##########################
# Veri Seti Hikayesi
##########################
"""
# UserId       :  Müşteri numarası
# ServiceId    : Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId   : Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate   : Hizmetin satın alındığı tarih
"""

# Proje Görevleri
##################################
# Görev 1: Veriyi Hazırlama
##################################
# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)     # bütün satırları göster
pd.set_option('display.width', 500)  # sütunlar max 500 tane gösterilsin
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
################################################
# Adım 1: armut_data.csv dosyasını okutunuz.
###############################################
df_ = pd.read_csv(r"C:\Users\GunalHincal\Desktop\datasets for github\recommender\armut_data.csv")
df = df_.copy()
df.head()
df.shape  # (162523, 7)
##################################################
# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.
# Elde edilmesi gereken çıktı:
"""
UserId    ServiceId    CategoryId         CreateDate       Hizmet
25446           4             5     6.08.2017 16:11          4_5
22948          48             5     6.08.2017 16:12         48_5
10618           0             8     6.08.2017 16:13          0_8
7256            9             4     6.08.2017 16:14          9_4
25446          48             5     6.08.2017 16:16         48_5
"""
###########################################################################
# df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

df["Hizmet"] = ["_".join([str(col[1]), str(col[2])]) for col in df.values]
df.head(10)
df.dtypes  # değişken type bilgisi getirir
df.info()  # aynısını daha detaylı getirir
#####################################################################
# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb.)
# bulunmamaktadır. Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması
# gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir
# date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında
# yeni bir değişkene atayınız.
# Elde edilmesi gereken çıktı:
"""
UserId    ServiceId    CategoryId         CreateDate         Hizmet    New_Date          SepetID
25446             4             5    6.08.2017 16:11            4_5     2017-08    25446_2017-08
22948            48             5    6.08.2017 16:12           48_5     2017-08    22948_2017-08
10618             0             8    6.08.2017 16:13            0_8     2017-08    10618_2017-08
7256              9             4    6.08.2017 16:14            9_4     2017-08     7256_2017-08
25446            48             5    6.08.2017 16:16           48_5     2017-08    25446_2017-08
"""
####################################################################################################

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df["New_Date"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()
df["SepetID"] = ["_".join([str(col[0]), str(col[5])]) for col in df.values]
df.head(10)

###############################################################
# Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
################################################################
# Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.
"""
       Hizmet       0_8     10_9     11_11     12_7     13_11     14_7..
      SepetID
    0_2017-08         0        0         0        0         0        0..
    0_2017-09         0        0         0        0         0        0..
    0_2018-01         0        0         0        0         0        0..
    0_2018-04         0        0         0        0         0        1..
10000_2017-08         0        0         0        0         0        0..
"""
###################################################################

pivot_table_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
pivot_table_df.head()


##################################
# Adım 2: Birliktelik kurallarını oluşturunuz.
#####################################

frequent_itemsets = apriori(pivot_table_df.astype(bool), min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()


#############################
# Adım3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
##################################

# arl_recommender adında bir fonksiyon tanımladık buna argümanlar tanımladık: rules_df diyoruz,
# product_id yi giriyoruz öneri yapılmasını istediğimiz,
# rec_count=1 adında öneri sayısı diye bir argüman giriyoruz default olarak 1 öneri yaptık şimdilik
# lift değerine göre sıralama yapacağız (istediğinize göre sıralarsınız)
# ve bunu büyükten küçüğe doğru ascending=False argümanı girerek sıralayacağız
# sonra boş bir öneri listesi oluşturup öneri ürünlerimizi o listeye atacağız
# daha sonra da bu rule ları lifte göre sıraladık,
# sonra da return ile reccomendation_list imizi çağırıyoruz içine argüman olarak [0:rec_count] girdik
# bu listenin içine şu kadar eleman seç demiş oluyoruz yani

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 4)  # ['22_0', '25_0', '15_1', '13_11']


