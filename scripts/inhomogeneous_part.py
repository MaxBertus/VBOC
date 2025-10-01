import numpy as np


if __name__ == "__main__":

    macroContainer = []

    containerVar = np.empty((0, 6)) * np.nan
    print(f'ContainerVar shape: {containerVar.shape}')

    for _ in range(8):
        a = np.random.normal(size=(1,6))
        containerVar = np.vstack((containerVar, a))
    
    macroContainer.append(containerVar)
    containerVar = np.empty((0, 6)) * np.nan

    for _ in range(3):
        b = np.random.normal(size=(1,6))
        containerVar = np.vstack((containerVar, b))

    print(f'ContainerVar shape: {containerVar.shape}')

    macroContainer.append(containerVar)

    print(f'Dimensions of solution sections: {[x.shape for x in macroContainer]}, type: {type(macroContainer)}')

    np.save('test_macro_container.npy', np.array(macroContainer, dtype=object), allow_pickle=True)

    loaded = np.load('test_macro_container.npy', allow_pickle=True)

    print(f'Dimensions of solution sections: {[x.shape for x in loaded]}, type: {type(loaded)}')

    print(loaded[0])
    print(loaded[0][:,0])