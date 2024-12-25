from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class Metrics:
    @classmethod
    def get_metrics(cls, y_pred, y_true):
        return {'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'auc_roc': roc_auc_score(y_true, y_pred)}
        
