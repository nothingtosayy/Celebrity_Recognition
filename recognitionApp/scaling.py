from sklearn.preprocessing import StandardScaler
def scaling(data):
    return StandardScaler().fit_transform(data)