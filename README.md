Количество потоков задаётся локальной статической константой kThreadsNum (в функции KMeans(const Points& data, const size_t K)).
Есть папка out, куда я при тестировании сохранял файлы с результатами работ программ.
В папке Common немного общей логики.

Для того, чтобы собрать основную программу надо запустить:

	make -f KMeansMakefile
Для того, чтобы собрать программу-тест надо запустить:

	make -f TestMakefile
Запуск основной программы:

	kmeans number_of_clusters input_file output_file
Например:

	./kmeans.out 10 shad.learn out/just_threads_centroids_K=10.txt

Запуск теста:

	test_main input_file_with_test_data input_file_with_centroids output_file
Например:

	/test_main.out shad.test out/just_threads_centroids_K=10.txt out/just_threads_RES_K=10.txt


Формат вывода центроидов в файл:

	centroid_id:0 \t component \t component \t ... \t component
 
	...

	centroid_id:K-1 \t component \t component \t ... \t component
