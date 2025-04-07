# Laporan Proyek Machine Learning - Ezraliano Sachio Krisnadiva

## Project Overview
Project ini adalah project yang bertujuan untuk membuat revenue analysis coffee shop dengan menggunakan machine learning. Machine learning yang dibuat diharapkan dapat memprediksi revenue yang didapat dengan menggunakan beberapa  fitur seperti produk apa saja yang telah dijual, jenis produk apa yang mengalami penjualan terbanyak, dan berapa nilai pertumbuhan pendapatan yang telah terjadi. Owner Coffee Shop dapat menggunakan machine learning untuk melihat insight yang ada agar dapat membuat sebuah keputusan yang tepat untuk perkembangan bisnis.


**Rubrik/Kriteria Tambahan (Opsional)**:
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
1. Melakukan Growth Analysis bulanan dan mingguan dengan mengelompokan data berdasarkan year, month, dan week, kemudian melakukan perhitungan total pendapatan dan persentase pertumbuhan. Setelah melakukan perhitungan didapatkan sebuah visualisasi data berupa line chart yang dapat digunakan sebagai insight dalam memahami pola tren pertumbuhan.
2. Melakukan pengelompokan data berdasarkan kategori product dan dan tipe product lalu menghitung total pendapatan, rata-rata pendapatan dan jumlah transaksi. Selanjutnya membuat visualisasi diagram yang dapat menunjukan pendapatan dari masing-masing produk.
3. Mengelompokan data pendapatan berdasarkan lokasi distribusi penjualan. Lokasi distribusi penjualan terdiri dari Astoria, Lower Manhattan, dan Hell's Kitchen. Setelah melakukan pengelompokan data berdasarkan lokasi, akan menghasilkan sebuah visualisasi distribusi penjualan berdasarkan lokasi berupa pie chart yang dapat digunakan sebagai insight untuk mengetahui lokasi distribusi yang memiliki penjualan terbesar dan yang terkecil.

**Rubrik/Kriteria Tambahan (Opsional)**:
    ### Solution statements
1. Prediksi revenue dengan model elastic net regression dengan metrik evaluasi (best_parameters, training_mse, testing_mse, Training R², Testing R², Cross-Validation R² Mean )
2. Identifikasi Produk Laris dan Rekomendasi dengan model linear regression dengan metrik evaluasi (Training MSE, Testing MSE, Training R² Score, Testing R² Score).
3. Visualisasi Pie Chart untuk mengetahui pendapatan per lokasi

## Data Understanding
Dataset kedai kopi Maven Roasters yang berbasis di New York City yang memiliki 3 lokasi berbeda dalam mendistribusikan penjualan. Informasi yang terdapat pada dataset ini mencakupi tanggal transaksi, waktu, lokasi geografis, hingga detail produk. Terdapat  149115 data yang ada pada dataset kedai kopi Maven Roaster, data tersebut tidak memiliki missing values maupun nilai duplicate. Hal ini menandakan dataset tersebut lumayan bersih hanya saja perlu dilakukan data prepocesing untuk mengatasi outliers dan skewness pada dataset ini.
Sumber[Kaggle](https://www.kaggle.com/datasets/agungpambudi/trends-product-coffee-shop-sales-revenue-dataset/data).

Variabel-variabel pada kedai kopi Maven Roaster adalah sebagai berikut:
- transaction_id : merupakan id transaksi
- transaction_date : merupakan tanggal terjadinya transaksi (format YYYY-MM-DD)
- transaction_time : merupakan waktu transaksi (format HH:MM:SS)
- transaction_qty : merupakan Jumlah produk yang dibeli dalam suatu transaksi.
- store_id : merupakan pengidentifikasi unik untuk setiap lokasi toko.
- store_location : merupakan nama atau deskripsi lokasi fisik toko
- product_id : 	merupakan pengidentifikasi unik untuk setiap produk yang dijual.
- unit_price : merupakan harga satu unit produk dalam transaksi.
- product_category : merupakan kategori umum tempat produk tersebut berada (misalnya, Kopi, Teh, Cokelat Minum).
- product_type : merupakan jenis atau varian produk tertentu (misalnya, Kopi seduh gourmet, Teh chai seduh, Cokelat panas).
- product_detail : merupakan Rincian tambahan tentang produk (misalnya, rasa, ukuran, atau campuran tertentu)

**Rubrik/Kriteria Tambahan (Opsional)**:
Dalam memahami dataset, saya menggunakan EDA yang berfokus pada visualisasi data. Berikut merupakan visualisasi data yang telah saya lakukan :
1. Visualisasi dalam menganalisis distribusi data yang digunakan untuk mengetahui skewness pada data.
2. Visualisasi Boxplot yang digunakan untuk mendeteksi outliers pada data
3. Visualisasi Heatmap yang digunakan untuk mengetahui hubungan korelasi antar variabel.
4. Visualisasi Barchart Revenue per kategori yang digunakan untuk memberikan insight terhadap total revenue dari masing-masing kategori product.
5. Visualisasi Barchart Revenue per tipe product yang digunakan untuk memberikan insights terhadap total revenue dari masing-masing tipe product.
6. Visualisasi Pie Chart Distribusi total Revenue berdasarkan Lokasi Toko yang digunakan untuk mengetahui toko mana yang memiliki pendapatan tertinggi dan terendah.
7. Visualisasi Line Chart yang digunakan untuk Mengetahui tren penjualan bulanan dan mingguan serta digunakan untuk mengetahui pertumbuhan dari penjualan berdasarkan waktu (bulan, minggu).


## Data Preparation
1. Load Dataset
2. Mengecek tipe data
3. Menganalisis distribusi Data untuk mengetahui skewness
4. Visualisasi skewness
5. Mengecek missing values
6. Menangani outliers
7. Heatmap Korelasi
8. Encoding
9. Menghitung Revenue
10. Menganalisis Growth Revenue

**Rubrik/Kriteria Tambahan (Opsional)**: 
1. Load dataset diperlukan untuk menyiapkan data ke dalam memori
2. Mengecek tipe data diperlukan untuk mengetahui tipe-tipe data apa saja yang ada di dalam dataset, dari sini dapat diketahui tipe data mana yang bertipe object/categorical dan tipe data mana saja yang bertipe numerik. 
3. Menganalisis distribusi data diperlukan untuk mengetahui pola distribusi di dalam dataset. dalam proses ini dilakukan perhitungan skewness dengan memilih kolom numerik untuk mengetahui apakah kolom tersebut memiliki skewness positif, netral, atau negatif. 
4. Visualisasi skewness diperlukan untuk melihat gambaran penyebaran distribusi data pada kolom numerik apakah kolom tersebut mengalami right skew, Symetric, atau left skew. Hasil yang didapatkan kebanyakan distribusi data mengalami symetric dan left skew.
5. Mengecek missing values dilakukan dengan cara mengecek keselruhan isi dari dataframe untuk mendeteksi apakah terdapat nilai null/nan pada kolom dataset. mengecek missing value diperlukan agar kualitas data yang diberikan dapat baik sehingga ketika dijalankan di model machine learning dapat memberikan nilai akurasi dan peforma model yang maksimal.
6. Menangani outliers dilakukan dengan cara visualisasi boxplot untuk kolom tipe data numerik, hal ini dilakukan agar mengetahui apakah data tersebut memiliki outliers yang bisa ditoleransi atau harus ditangani lebih lanjut. Jika outliers tidak dapat ditoleransi maka akan dilakukan proses penghapusan outliers menggunakan IQR.
7. Heatmap korelasi dilakukan dengan cara membuat matrix heatmap korelasi, matrix ini digunakan untuk melihat kekuatan korelasi diantara variabel-variabel yang ada.
8. Encoding dilakukan dengan cara memilih kolom data yang bertipe object setelah ini kolom tersebut diubah tipe datanya menjadi numerik dengan metode one hot encoding menggunakan pd.get_dummies(). Hal ini diperlukan karena model machine learning tidak dapat memproses data yang memiliki tipe data object dan hanya bisa memproses data dengan tipe data numeric.
9. Menghitung Revenue diperlukan untuk mengetahui hasil penjualan dari masing-masing produk, Untuk menghitung revenue dapat menggunakan rumus transaction_qty * unit_price. setelah kalkulasi revenue selanjutnya akan dilakukan sebuah visualisasi agar dapat mengetahui produk-produk mana saja yang paling laris terjual dan wilayah mana yang berkontribusi besar dalam penjualan. Terdapat 3 macam visualisasi yang pertama visualisasi total revenue per kategori produk berupa bar chart, kedua visualisasi revenue per tipe produk yang berupa bar chart, dan yang ketiga visualisasi Distribusi total revenue berdasarkan lokasi toko berupa pie chart.
10. Menganalisis growth revenue diperlukan untuk mengetahui tingkat pertumbuhan penjualan dari waktu ke waktu, hal ini dapat memberikan gambaran/insight yang dapat digunakan untuk menganalisis pola tren musiman penjualan. Terdapat 4 tipe analisa growth pada project ini yang pertama analisa growth bulanan untuk mengetahui tren penjualan bulanan yang berupa line chart, yang kedua analisa growth mingguan untuk mengetahui tren penjualan mingguan yang berupa line chart,  yang ketiga analisa growth bulanan per kategori produk yang dapat digunakan untuk memberikan insight produk mana saja yang mengalami pertumbuhan penjualan terbesar yang berupa bar chart, dan yang keempat analisa growth bulanan per lokasi toko yang digunakan untuk mengetahui pertumbuhan penjualan berdasarkan distribusi toko yang berupa bar chart.


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


**Rubrik/Kriteria Tambahan (Opsional)**: 
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
1. Mean Squared Error (MSE) : MSE digunakan untuk mengukur rata-rata kuadrat kesalahan antara nilai aktual dan nilai prediksi. Semakin kecil nilainya, semakin baik modelnya.
2. R² : R² digunakan untuk mengukur proporsi variansi dalam data target yang dapat dijelaskan oleh model. Nilainya berkisar dari 0 hingga 1 (terkadang negatif jika model sangat buruk).
3. Cross-Validation untuk model Elastic Net : digunakan untuk mengukur stabilitas dan performa model dengan membagi data menjadi beberapa lipatan (folds), melatih pada sebagian data, dan menguji pada sisanya, lalu mengambil rata-rata hasilnya.
pada hasil pelatihan model, didapatkan hasil evaluasi sebagai berikut :
- Model Elastic Net 
Training MSE: 0.21
Testing MSE: 0.22
Training R²: 0.9452
Testing R²: 0.9438
- Model Linear Regression
Training MSE: 0.17
Testing MSE: 0.18
Training R²: 0.9554
Testing R²: 0.9550

Dari sini dapat disimpulkan bahwa model linear regression memiliki performa yang jauh lebih baik.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Formula metrik evaluasi untuk MSE adalah sebagai berikut
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



**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
