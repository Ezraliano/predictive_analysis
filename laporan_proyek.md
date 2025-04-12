# Laporan Proyek Machine Learning - Ezraliano Sachio Krisnadiva

## Project Overview
Project ini adalah project yang bertujuan untuk membuat revenue analysis coffee shop dengan menggunakan machine learning. Machine learning yang dibuat diharapkan dapat memprediksi revenue yang didapat dengan menggunakan beberapa  fitur seperti produk apa saja yang telah dijual, jenis produk apa yang mengalami penjualan terbanyak, dan berapa nilai pertumbuhan pendapatan yang telah terjadi. Owner Coffee Shop dapat menggunakan machine learning untuk melihat insight yang ada agar dapat membuat sebuah keputusan yang tepat untuk perkembangan bisnis.
Dalam mendukung proses proyek ini berikut merupakan alasan mengapa proyek ini perlu diselesaikan dan terdapat refrensi riset pendukung untuk proyek ini :
- Proyek ini perlu diselesaikan karena owner sangat terbantu dengan insight yang diberikan oleh machine learning. karena machine learning dapat memprediksi revenue berdasarkan product, jenis product dan lokasi coffee shop yang ada serta dapat melihat pertumbuhan penjualan yang dapat digunakan oleh owner coffee shop untuk melihat pola musiman penjualan.
- Refrensi
  [Site Selection Prediction for Coffee Shops Based on Multi-Source Space Data Using Machine Learning Techniques](https://www.mdpi.com/2220-9964/12/8/329) Author by ( Jianqi Zhao, Baiyi Zhong, Ling Wu)

## Business Understanding
Coffee shop merupakan bisnis di sektor ritel yang bergantung pada penjualan produk minuman. Produk minuman yang biasa dijual di coffee shop meliputi kopi, teh, makanan ringan dan minuman lainnya di berbagai lokasi toko. Dalam lingkungan bisnis yang kompetitif, pemahaman mendalam tentang performa penjualan, tren pendapatan, dan faktor pendorong pertumbuhan menjadi kunci untuk tetap bersaing secara relevan dan menerima profit. Data pendapatan (revenue) yang mencakup informasi transaksi, ketegori produk, tipe produk, dan lokasi toko dapat memberikan insight yang dapat meningkatkan strategi operasional dan pemasaran. Dari uraian di atas terdapat beberapa problem statements sebagai berikut:

### Problem Statements
1. Bagaimana tren pendapatan dari waktu ke waktu, dan kapan periode penjualan puncak dan penurunan terjadi?
2. Produk apa yang paling laris terjual di pasaran?
3. Lokasi mana yang menyumbang revenue yang terbesar dan yang terkecil?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
1. Melakukan Growth Revenue Analysis bulanan dengan mengelompokan data berdasarkan year, month, dan week, kemudian melakukan perhitungan total pendapatan dan persentase pertumbuhan. Setelah melakukan perhitungan didapatkan sebuah visualisasi data berupa line chart yang dapat digunakan sebagai insight dalam memahami pola tren pertumbuhan.
2. Melakukan pengelompokan data berdasarkan kategori product dan dan tipe product lalu menghitung total pendapatan, rata-rata pendapatan dan jumlah transaksi. Selanjutnya membuat visualisasi diagram yang dapat menunjukan pendapatan dari masing-masing produk.
3. Mengelompokan data pendapatan berdasarkan lokasi distribusi penjualan. Lokasi distribusi penjualan terdiri dari Astoria, Lower Manhattan, dan Hell's Kitchen. Setelah melakukan pengelompokan data berdasarkan lokasi, akan menghasilkan sebuah visualisasi distribusi penjualan berdasarkan lokasi berupa pie chart yang dapat digunakan sebagai insight untuk mengetahui lokasi distribusi yang memiliki penjualan terbesar dan yang terkecil.

    ### Solution statements
1. Prediksi revenue dengan model elastic net regression dengan metrik evaluasi (best_parameters, training_mse, testing_mse, Training R², Testing R², Cross-Validation R² Mean )
2. Identifikasi Produk Laris dan Rekomendasi dengan model linear regression dengan metrik evaluasi (Training MSE, Testing MSE, Training R² Score, Testing R² Score).
3. Visualisasi Pie Chart untuk mengetahui pendapatan per lokasi

## Data Understanding
Dataset kedai kopi Maven Roasters yang berbasis di New York City yang memiliki 3 lokasi berbeda dalam mendistribusikan penjualan. Informasi yang terdapat pada dataset ini mencakupi tanggal transaksi, waktu, lokasi geografis, hingga detail produk. Terdapat  149115 data yang ada pada dataset kedai kopi Maven Roaster, data tersebut tidak memiliki missing values maupun nilai duplicate. Hal ini menandakan dataset tersebut lumayan bersih hanya saja perlu dilakukan data prepocesing untuk mengatasi outliers dan skewness pada dataset ini.
Sumber[Kaggle](https://www.kaggle.com/datasets/agungpambudi/trends-product-coffee-shop-sales-revenue-dataset/data).

Berikut Informasi Umum terkait Dataset :
- Jumlah Data : 149115 transaksi
- Jumlah Kolom : 11 

Kondisi Data :
1. Missing Values : Tidak ada nilai yang hilang
2. Data Duplikat : Tidak ada data duplikat
3. Outliers : Terdapat outlier pada kolom numerik seperti transaction_qty dan unit_price, yang diidentifikasi menggunakan metode IQR.
4. Skewness : Kolom unit_price menunjukkan distribusi yang miring ke kanan (positively skewed), yang kemudian diatasi dengan transformasi logaritmik (unit_price_log) untuk mendekati distribusi normal.


Berikut merupakan uraian fitur yang ada pada dataset:
- transaction_id : Pengidentifikasi unik untuk setiap transaksi (tipe: integer).
- transaction_date : Tanggal transaksi dalam format YYYY-MM-DD (tipe: object)
- transaction_time : Waktu transaksi dalam format HH:MM:SS (tipe: object).
- transaction_qty : Jumlah produk yang dibeli dalam satu transaksi (tipe: integer).
- store_id : Pengidentifikasi unik untuk setiap lokasi toko (tipe: integer).
- store_location : Nama atau deskripsi lokasi toko (tipe: object).
- product_id : 	Pengidentifikasi unik untuk setiap produk (tipe: integer).
- unit_price : Harga satuan produk dalam transaksi (tipe: float).
- product_category : Kategori umum produk, seperti "Coffee", "Tea", atau "Drinking Chocolate" (tipe: object)
- product_type : Varian spesifik dalam kategori, misalnya "Gourmet brewed coffee", "Chai tea", atau "Hot chocolate" (tipe: object).
- product_detail : Informasi tambahan tentang produk, seperti rasa, ukuran, atau campuran (misalnya, "Ethiopian Large" atau "Decaf Espresso") (tipe: object).


## Proses Exporasi Data Analysis (EDA)
Dalam memahami dataset, saya menggunakan EDA untuk membantu proses explorasi dan memahami data. Berikut adalah beberapa insight penting yang diperoleh dari dataset: :

1. Distribusi Data Numerik dengan visualisasi diagram batang :
- Kolom transaction_qty memiliki distribusi yang cenderung simetris, dengan sebagian besar transaksi melibatkan 1-2 item. Namun, terdapat beberapa transaksi dengan jumlah besar (outlier), yang mungkin mencerminkan pembelian grosir atau pesanan khusus.
- Kolom unit_price menunjukkan skewness positif, dengan banyak produk memiliki harga rendah (misalnya, kopi atau teh standar) tetapi beberapa produk premium memiliki harga lebih tinggi (misalnya, specialty coffee).

2. Outliers :
- Outlier pada transaction_qty dan unit_price diidentifikasi menggunakan metode IQR dan visualisasi boxplot.

3. Korelasi :
- Heatmap korelasi menunjukkan bahwa transaction_qty dan unit_price memiliki korelasi rendah, menandakan bahwa jumlah item yang dibeli tidak selalu berkaitan dengan harga satuan produk.


## Data Preparation
Tahap Data Preparation bertujuan untuk mempersiapkan dataset agar siap digunakan untuk analisis lanjutan dan pemodelan machine learning. Proses ini mencakup pembersihan data, transformasi data, pembuatan fitur baru, dan pemrosesan data lainnya untuk memastikan kualitas dan kompatibilitas data dengan model. Berikut merupakan tahapan dari proses Data Preparation :
1. Penanganan Skewness
2. Penanganan outliers dengan metode IQR
3. Proses Encoding kategorikal dengan menggunakan metode one-hot encoding menggunakan pd.get_dummies()
4. Menerapkan Feature Engineering proses penambahan fitur revenue
5. Proses Standarisasi Fitur numerik
6. Proses Split Data


## Penjelasan lebih lanjut tentang proses Data Preparation: 
1. Penanganan Skewness :
- Melakukan log transformation pada kolom unit_price menggunakan np.log1p untuk menghasilkan kolom baru unit_price_log. Skewness dihitung sebelum dan sesudah transformasi untuk memverifikasi bahwa distribusi menjadi lebih normal. 
- Mengapa dilakukan penanganan skewness? karena kolom unit_price memiliki distribusi yang condong (skewed), yang dapat memengaruhi performa model regresi. Transformasi logaritma mengurangi skewness, membuat data lebih simetris dan mendekati distribusi normal.
2. Penanganan outliers dengan metode IQR
-  Menghapus outlier dari kolom numerik (transaction_qty, unit_price.) menggunakan metode IQR. Data yang berada di luar batas bawah (Q1 - 1.5*IQR) atau batas atas (Q3 + 1.5*IQR) dihapus.
- Mengapa dilakukan penanganan outliers dengan metode IQR? karena outlier dapat menyebabkan model regresi memberikan prediksi yang bias atau tidak akurat karena sensitivitasnya terhadap nilai ekstrem. Penghapusan outlier menggunakan IQR adalah metode yang robust untuk memastikan data berada dalam rentang yang wajar, meningkatkan stabilitas model.
3. Proses Encoding kategorikal dengan menggunakan metode one-hot encoding menggunakan pd.get_dummies()
- Setiap kategori dalam kolom kategorikal diubah menjadi kolom baru dengan nilai 0 atau 1, memungkinkan model untuk memproses data non-numerik.
- Mengapa dilakukan proses encoding kategorikal dengan menggunakan metode one-hot encoding menggunakan pd.get_dummies() ? karena Model machine learning seperti regresi tidak dapat langsung memproses data kategorikal dalam bentuk teks/object. One-Hot Encoding mengubah data kategorikal menjadi format numerik tanpa mengasumsikan urutan antar kategori, sehingga cocok untuk variabel seperti lokasi toko atau tipe produk.
4. Menerapkan Feature Engineering proses penambahan fitur revenue
- Fitur revenue dibuat sebagai target prediksi, sedangkan fitur waktu memungkinkan analisis tren pendapatan. Analisis tambahan seperti revenue per kategori, tipe produk, dan lokasi dilakukan untuk wawasan bisnis.
- Mengapa menerapkan Feature Engineering proses penambahan fitur revenue ? karena Fitur revenue adalah variabel target yang relevan untuk prediksi pendapatan, dihitung langsung dari data transaksi. 
5. Proses Standarisasi Fitur numerik
- Standarisasi mengubah fitur numerik agar memiliki mean=0 dan standar deviasi=1, memastikan semua fitur berada pada skala yang sama.
- Mengapa dilakukan Proses Standarisasi Fitur numerik? Karena Model regresi seperti Elastic Net dan Linear Regression sensitif terhadap skala fitur. Fitur dengan skala besar (misalnya, unit_price) dapat mendominasi fitur dengan skala kecil (misalnya, transaction_qty). Proses Standarisasi memastikan kontribusi setiap fitur seimbang, meningkatkan akurasi dan stabilitas model.
6. Proses Split Data
- Data pelatihan digunakan untuk melatih model, sedangkan data pengujian digunakan untuk mengevaluasi performa model pada data yang belum dilihat, porse split data adalah 80:20.
- Mengapa melakukan proses split Data ? karena pembagian data memungkinkan evaluasi model yang objektif, memastikan model dapat menggeneralisasi dengan baik pada data baru. Proporsi 80:20 adalah praktik umum untuk menyeimbangkan jumlah data pelatihan dan pengujian, dengan random_state memastikan hasil yang konsisten.

## Modeling
Terdapat 2 model yang digunakan untuk project ini antara lain :
1. Elastic Net Rergression
2. Linear Regression
Elastic Net Regression bertujuan untuk menggabungkan L1 (Lasso) dan L2 (Ridge) untuk mengatasi overfitting. Pada project ini Elastic Net Regression digunakan untuk memprediksi pendapatan(revenue).
Linear Regression dalam machine learning yang digunakan untuk memprediksi nilai kontinu berdasarkan satu atau lebih fitur input. Pada project ini linear regression digunakan untuk memprediksi pendapatan(revenue).
Pada pelatihan model terdapat Pemilihan fitur (X) dan target (y) sebagai berikut :
X berisi ('transaction_date', 'transaction_id', 'transaction_time', 'store_id', 'product_id', 'year', 'month', 'week', 'day')
y berisi ('revenue')
Pembagian data train dan data test dengan komposisi 80% dan 20%.


## Penjelasan Lebih Lanjut tentang model: 
1. Elastic Net Regression memiliki performa yang baik dalam melakukan prediksi revenue. Elastic net regression menggunakan optimasi gradient descent untuk menemukan bias yang menyeimbangkan kesalahan prediksi penalti. Model Elastic net Regression memiliki beberapa kelebihan dan kekuarangan antara lain :
Kelebihan :
- Menangani Multikolinearitas
- Dapat mengeliminasi fitur tidak relevan (dari L1).
- Mengurangi overfitting dengan regularisasi (dari L2).
- Kombinasi L1 dan L2 memungkinkan penyesuaian sesuai kebutuhan data.
Kekurangan :
- Lebih Kompleks
- Memerlukan Tuning
- Kurang Efektif pada Data Sederhana
2. Linear Regression memiliki performa yang jauh lebih unggul pada project ini, hal ini dibuktikan dengan hasil evaluasi MSE dan R² yang menunjukan linear regression jauh lebih baik. Linear Regression menggunkan metode Least Squares untuk menemukan bias yang meminimalkan total kuadrat kesalahan antara prediksi dan nilai aktual. Model linear regression memiliki beberapa kelabihan dan kekurangan antara lain :
Kelebihan :
- Sederhana dan Interpretabel
- Komputasi ringan karena tidak ada parameter tambahan.
- Efektif pada Data Linier
Kekurangan :
- Sensitif terhadap Multikolinearitas
- Rentan Overfitting
- Tidak Ada Seleksi Fitur

## Evaluation
Terdapat beberapa metrik evaluasi yang digunakan untuk melakukan evaluasi model, antara lain :
1. Mean Squared Error (MSE)
- MSE mengukur rata-rata kuadrat selisih antara nilai aktual dan prediksi, memberikan indikasi seberapa besar kesalahan prediksi model.
- Relevansi: Dalam konteks regresi untuk memprediksi revenue (nilai kontinu), MSE relevan karena fokus pada minimisasi kesalahan prediksi dalam satuan kuadrat revenue. Nilai MSE yang lebih kecil menunjukkan prediksi yang lebih akurat, yang penting untuk memberikan estimasi pendapatan yang dapat dipercaya kepada owner coffee shop.
- - Formula metrik evaluasi untuk MSE adalah sebagai berikut
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
keterangan :
- yᵢ: Nilai aktual.
- ŷᵢ: Nilai prediksi.
- n: Jumlah data.
Cara kerja MSE :
1. Menghitung selisih Untuk setiap data point, kurangi nilai prediksi (ŷᵢ) dari nilai aktual (yᵢ)
2. Melakukan kuadrat selisih dan hasilnya untuk menghilangkan tanda negatif dan menekankan kesalahan besar.
3. Menambahkan semua kuadrat selisih
4. Menghitung rata-rata dengan jumlah data (n)

Berikut merupakan implementasi MSE yang ada pada source code
- Elastic Net Regression
train_mse_enet = mean_squared_error(y_train, y_train_pred_enet)
test_mse_enet = mean_squared_error(y_test, y_test_pred_enet)
Keterangan :
- y_train dan y_test: Revenue aktual dari data pelatihan dan pengujian.
- y_train_pred_enet dan y_test_pred_enet: Revenue prediksi dari Elastic Net.
- Fungsi mean_squared_error melakukan perhitungan di atas secara otomatis untuk semua data.

- Linear Regression
train_mse_linear = mean_squared_error(y_train, y_train_pred_linear)
test_mse_linear = mean_squared_error(y_test, y_test_pred_linear)

2. R² Score:
- R² mengukur proporsi variansi dalam data target (revenue) yang dapat dijelaskan oleh model, dengan nilai antara 0 hingga 1 (atau negatif jika model sangat buruk).
- Relevansi: R² relevan karena memberikan gambaran seberapa baik model menangkap pola dalam data transaksi coffee shop. Nilai R² yang mendekati 1 menunjukkan bahwa model dapat menjelaskan sebagian besar variabilitas revenue, yang penting untuk memastikan prediksi mendukung pengambilan keputusan bisnis.
- Formula Metrik R² Score
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$
Keterangan :
**\( SS_{res} \)** (Sum of Squared Residuals): Jumlah kuadrat selisih antara nilai aktual dan nilai prediksi  
  $$
  SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

- **\( SS_{tot} \)** (Total Sum of Squares): Jumlah kuadrat selisih antara nilai aktual dan rata-rata nilai aktual  
  $$
  SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2
  $$

- **\( y_i \)**: Nilai aktual  
- **\( \hat{y}_i \)**: Nilai prediksi  
- **\( \bar{y} \)**: Rata-rata dari semua nilai aktual  
- **\( n \)**: Jumlah data


Cara Kerja R²
1. Menghitung rata-rata dari semua nilai aktual
2. Menjumlahkan kuadrat selisih antara setiap nilai aktual dan rata-rata.
3. Menjumlahkan kuadrat selisih antara nilai aktual dan prediksi.
4. Menghitung Rasio dari SS res dan SS tot.
5. Mengurangi dari 1 R² = 1 - rasio

Berikut merupakan implementasi dari R²
- Elastic Net Regression
train_r2_enet = r2_score(y_train, y_train_pred_enet)
test_r2_enet = r2_score(y_test, y_test_pred_enet)

- Linear Regression
train_r2_linear = r2_score(y_train, y_train_pred_linear)
test_r2_linear = r2_score(y_test, y_test_pred_linear)


Alasan Pemilihan Metrik MSE dan R² adalah karena proyek ini adalah masalah regresi, di mana fokusnya adalah memprediksi nilai revenue secara akurat dan menjelaskan variabilitas data. MSE memberikan ukuran kesalahan absolut, sedangkan R² memberikan konteks proporsional tentang kecocokan model.

Hasil Evaluasi :
1. Elastic Net Regression:
- Training MSE: 0.21
- Testing MSE: 0.22
- Training R²: 0.9452
- Testing R²: 0.9438

2. Linear Regression :
- Training MSE: 0.17
- Testing MSE: 0.18
- Training R²: 0.9554
- Testing R²: 0.9550

Dari sini dapat dilihat Model Linear Regression memiliki performa yang lebih baik dibandingkan Elastic Net, dengan MSE yang lebih rendah (0.17 pada pelatihan dan 0.18 pada pengujian) dan R² yang lebih tinggi (sekitar 0.955). Ini menunjukkan bahwa model dapat memprediksi revenue dengan kesalahan lebih kecil dan menjelaskan sekitar 95.5% variabilitas data. Selisih kecil antara metrik pelatihan dan pengujian menunjukkan generalisasi yang baik.

Komparasi dan Penentuan Model Terbaik
Komparasi :
- MSE: Linear Regression memiliki MSE yang lebih rendah (0.17 vs 0.21 pada pelatihan, 0.18 vs 0.22 pada pengujian) dibandingkan Elastic Net, menandakan prediksi yang lebih akurat.
- R²: Linear Regression memiliki R² yang lebih tinggi (0.9554 vs 0.9452 pada pelatihan, 0.9550 vs 0.9438 pada pengujian), menunjukkan kemampuan yang lebih baik dalam menjelaskan variabilitas revenue.
- Kompleksitas: Elastic Net menggunakan regularisasi (L1 dan L2) untuk menangani multikolinearitas dan seleksi fitur, yang mungkin relevan jika data memiliki banyak fitur redundan. Namun, performa Linear Regression yang lebih baik menunjukkan bahwa hubungan antara fitur dan revenue cenderung linier dan tidak memerlukan regularisasi tambahan.

Model Terbaik : Berdasarkan MSE dan R², Linear Regression adalah model terbaik untuk proyek ini. Model ini memberikan prediksi revenue yang lebih akurat dan menjelaskan variabilitas data dengan lebih baik dibandingkan Elastic Net.

Hasil evaluasi model memiliki dampak signifikan terhadap Business Understanding dan menjawab Problem Statements, Goals, dan Solution Statements sebagai berikut:
1. Problem Statements:
- Tren pendapatan dari waktu ke waktu, periode puncak, dan penurunan : Linear Regression dengan R² 0.955 memungkinkan prediksi revenue yang akurat berdasarkan fitur seperti transaction_qty, unit_price, dan unit_price_log. Meskipun fitur waktu (year, month, week, day) tidak digunakan dalam model akhir, analisis feature engineering (growth revenue bulanan) telah menghasilkan visualisasi line chart yang menunjukkan tren pendapatan dan pola musiman. 
- Dampak : Owner coffee shop dapat menggunakan prediksi revenue untuk mengantisipasi periode puncak (misalnya, akhir tahun) dan penurunan, memungkinkan perencanaan stok dan promosi iklan yang lebih baik.

- Produk apa yang paling laris terjual di pasaran: Analisis feature engineering menghasilkan barplot revenue berdasarkan product_category dan product_type, mengidentifikasi produk laris seperti kopi atau teh. 
- Owner dapat memprioritaskan stok dan pemasaran untuk produk laris, meningkatkan efisiensi operasional dan pendapatan.

- Lokasi mana yang menyumbang revenue terbesar dan terkecil: Visualisasi pie chart dari feature engineering menunjukkan distribusi revenue berdasarkan store_location (Astoria, Lower Manhattan, Hell's Kitchen). Model Linear Regression menggunakan store_location sebagai fitur kategorikal, memungkinkan prediksi revenue yang mempertimbangkan perbedaan lokasi.

2. Goals :
- Growth Revenue Analysis bulanan: Analisis feature engineering menghasilkan line chart revenue bulanan dengan persentase pertumbuhan, memberikan insight tentang tren dan pola musiman. Model Linear Regression mendukung prediksi revenue yang akurat, memvalidasi tren ini dengan akurasi tinggi (R² 0.955).
- Dampak : Owner dapat merencanakan strategi jangka panjang berdasarkan pola musiman, seperti meningkatkan promosi selama periode puncak.

- Visualisasi pendapatan per kategori dan tipe produk: Barplot revenue berdasarkan product_category dan product_type memberikan gambaran jelas tentang produk yang menghasilkan pendapatan tertinggi. Model Linear Regression memanfaatkan fitur ini untuk prediksi, memastikan bahwa insight ini dapat dipercaya.
- Dampak : Owner dapat mengoptimalkan menu dengan fokus pada kategori atau tipe produk yang paling menguntungkan.

- Visualisasi distribusi pendapatan per lokasi : Pie chart distribusi revenue berdasarkan store_location menunjukkan kontribusi masing-masing lokasi. Model Linear Regression menggunakan store_location untuk prediksi, memastikan bahwa perbedaan antar lokasi dipertimbangkan.
- Dampak : Owner dapat membuat keputusan berbasis lokasi, seperti ekspansi di lokasi yang menguntungkan atau perbaikan di lokasi yang kurang perform.

3. Solution Statements :
- Prediksi revenue dengan Elastic Net Regression : Walaupun Elastic Net memberikan performa yang baik (R² 0.9438), model ini kurang optimal dibandingkan Linear Regression. Namun, regularisasi dalam Elastic Net dapat membantu dalam skenario dengan data yang lebih kompleks di masa depan, memastikan stabilitas prediksi. Solusi ini mendukung prediksi revenue yang akurat, tetapi Linear Regression lebih efektif untuk kebutuhan saat ini.
- Identifikasi produk laris dan rekomendasi dengan Linear Regression: Linear Regression dengan R² 0.955 memungkinkan prediksi revenue yang sangat akurat, mendukung analisis produk laris melalui fitur seperti product_category dan product_type. Rekomendasi seperti fokus pada produk tertentu dapat dipercaya karena akurasi model yang tinggi. Solusi ini berhasil memberikan insight yang actionable untuk optimasi menu dan strategi pemasaran.
- Pie chart memberikan gambaran visual yang jelas tentang distribusi revenue, yang divalidasi oleh model Linear Regression yang menggunakan store_location sebagai fitur. Ini memungkinkan owner untuk membuat keputusan berbasis data, seperti alokasi sumber daya antar lokasi. Solusi ini berhasil menjawab pertanyaan tentang lokasi dengan revenue terbesar dan terkecil, mendukung strategi operasional.

## Kesimpulan
1. Model Linear Regression dan analisis feature engineering secara efektif menjawab ketiga problem statements dengan memberikan prediksi revenue yang akurat, mengidentifikasi produk laris melalui visualisasi, dan menunjukkan distribusi pendapatan per lokasi.
2. Ketiga tujuan (analisis tren, visualisasi produk, distribusi lokasi) tercapai melalui kombinasi model dan visualisasi, memberikan insight yang actionable untuk owner coffee shop.
3. Prediksi revenue dengan Linear Regression adalah solusi paling berdampak karena akurasinya yang tinggi (R² 0.955), mendukung pengambilan keputusan strategis. Identifikasi produk laris dan visualisasi lokasi juga berdampak signifikan dengan memberikan panduan operasional dan pemasaran yang jelas.






