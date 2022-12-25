from matplotlib import pyplot as plt

class ChartBuilder2:
    @staticmethod
    def build_training_data_chart(accepted_df, rejected_df):
        plt.plot(accepted_df['test1'], accepted_df['test2'], '*', label='Accepted')
        plt.plot(rejected_df['test1'], rejected_df['test2'], 'x', label='Rejected')
        plt.legend(loc=3)
        plt.grid()
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')

        plt.show()

    @staticmethod
    def build_training_3d(data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data.test1, data.test2, data.is_accepted)

        plt.show()
