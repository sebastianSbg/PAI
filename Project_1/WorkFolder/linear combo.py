    bestModel = Model()
    highestScore = 0
    for a in A:
        kernel = ConstantKernel() * a*RBF() + ConstantKernel()
        M = Model(kernel)
        M.fit_model(train_x, train_y)

        # predcit on test set
        prediction = M.predict(train_x)
        cost = cost_function(prediction,train_y)
        print("Cost with a = " + str(a) + ", : " + str(cost))    
        if cost < highestScore:
            highestScore = cost  