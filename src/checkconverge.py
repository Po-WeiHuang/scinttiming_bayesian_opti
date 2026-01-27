#check reaching convergence or not

# justify coverging or not
from pathlib import Path
import numpy as np

import  os 

print("PWD:",os.getcwd())


if __name__ == "__main__":
    with open("algo_log.txt", "a") as outlog:
        outlog.write("\n\n")
        outlog.write("### THIS IS CHECKCONVERGE SCRIPT ###\n")
        dir_path = Path("results/pars")
        arrays = [np.load(p) for p in dir_path.glob(f"*.npy")]

        '''
        # Test Example
        arr1 = np.array([
            1.0, 130.0, 300.0, 500.0,
            1.0,
            0.90, 0.05, 0.025, 0.025,
            0,
            7550.0,
            1705233000
        ])

        arr2 = np.array([
            1.2, 135.0, 310.0, 520.0,
            1.0,
            0.88, 0.06, 0.03, 0.03,
            1,
            7800.0,
            1705233060
        ])

        arr3 = np.array([
            0.9, 128.0, 295.0, 490.0,
            1.1,
            0.92, 0.04, 0.02, 0.02,
            2,
            7700.0,
            1705233120
        ])

        arr4 = np.array([
            1.1, 140.0, 320.0, 540.0,
            0.95,
            0.85, 0.07, 0.04, 0.04,
            3,
            7800.0,
            1705233180
        ])

        arr5 = np.array([
            1.0, 125.0, 280.0, 470.0,
            1.05,
            0.93, 0.035, 0.02, 0.015,
            4,
            7600.0,
            1705233240
        ])
        train = np.vstack([arr1,arr2,arr3,arr4,arr5])

        '''
        train = np.vstack(arrays)   # shape (N, D)
        print(train.shape)
        # train_sort will be order by chi2 with descending order 
        train_sort = np.array([train[i] for i in np.argsort(train[:,-2])])

        # check in covergence
        # define covergence if
        # compared the minimum of chi2
        # other 3 points whose
        # chi2_min < 1500 +
        # t1,A1 are within 5%
        chi2_min = train_sort[0,-2]
        print("chi2_min ",chi2_min)
        best_par = train_sort[0,:-3]
        par_chi2small = train_sort[:4,:-3] 
        all_True = False
        epsilon = 10e-6
        iterations = int(np.max(train_sort[:,-3]))

        if iterations <= 5 or chi2_min > 1500: pass
        else:
            shift = np.abs(par_chi2small - best_par)/(best_par+epsilon)
            #print(shift)
            outlog.write(f"\nPercentage shift:{shift}")
            shift_t1A1 = shift[:,[0,5]]  
            #print(shift_t1A1)
            all_True = np.all(shift_t1A1 <= 0.05) 


        print("iterations ",iterations)
        
        #print(all_True)
        if all_True == True:
            outlog.write("\n*********** Results converge **************")
            print("*********** Results converge **************")
            print(f"Best par set is {best_par}")
            outlog.write(f"\nBest par set is {best_par}")
            exit(0)
        else:
            print("*********** Results not converge yet **************")
            outlog.write("\n*********** Results not converge yet **************")
            print("*********** Keep simulating later *******************")
            outlog.write("\n*********** Keep simulating later *******************")
            print(f"Current par set is {train_sort[0,:-3]}")
            outlog.write(f"\nCurrent par set is {train_sort[0,:-3]}")
            print(f"Best par set is {best_par}")
            outlog.write(f"\nBest par set is {best_par}")
            exit(1)


