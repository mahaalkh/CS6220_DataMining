I read the datasets from the paths ("datasets/dataset1.txt") , ("datasets/dataset2.txt") and ("datasets/dataset3.txt")
--------------------------------------------------------------------------------------------------------------------


- to run DBSCAN on the data call  
    DBSCAN(eps, min_points, input_matrix) 

- to run K_means: 
   k_means(K, input_matrix, initial_means = False, num_restarts = 100)
       output > cluster_assignments, cluster_means, sse 

- to run K_means and plot it : 
   plot_k_means(data_matrix, K, original_labels, clustering_text, num_restarts=100, initial_means=False)
       output > cluster_assignments, cluster_means, sse
   
- to run GMM: 
    GMM(data_matrix, K, num_restarts = 100, means = None, variances = None)
        output > means, variances, soft_assignments, log_likelihood
    
- to run GMM and plot it: 
     plot_GMM(data_matrix, K, original_labels, clustering_text, num_restarts=100, initial_means=None, initial_vars=None)
         output > means, variances, predicted_labels, log_likelihood
         
      
--------------------------------------------------------------------------------------------------------------------------\

The written part is in the python notebook with details in the PDF. 

Question three answers is in the last three pages of the PDF 
--------------------------------------------------------------------------------------------------------------------------
 
 Difficulties: 
   - Understanding Gaussian Mixure Models was not easy and the help of the professor for my understanding was extremely helpful
   
----------------------------------------------------------------------------------------------------------------------------

Time spent ~ 45 hrs (alot of time spent debugging and understanding GMM)

----------------------------------------------------------------------------------------------------------------------------

Discussed the HW with Sanil,  Jason and Timur
   
    