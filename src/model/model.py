import torch
import torch.nn as nn
import torch.nn.functional as F
from complex_layers import ComplexConvBlock, complex_magnitude
from attention import AttentionBlock


class ComplexAnomalyDetector(nn.Module):
    def __init__(self, num_classes=4, input_channels=1):
        super(ComplexAnomalyDetector, self).__init__()
        
        self.complex_layers = nn.ModuleList([
            ComplexConvBlock(input_channels, 45, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2)),
            ComplexConvBlock(45, 90, kernel_size=(7, 5), stride=(2, 2), padding=(3, 2)),
            ComplexConvBlock(90, 90, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            ComplexConvBlock(90, 90, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            ComplexConvBlock(90, 45, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0))
        ])
        
        self.magnitude_conv = nn.Conv2d(45, 4, kernel_size=(5, 1), stride=(2, 1), padding=(1, 0))
        
        self.complex_conv = nn.Conv2d(90, 4, kernel_size=(5, 1), stride=(2, 1), padding=(1, 0))
        
        self.attention = AttentionBlock(8)
        
        self.total_conv = nn.Conv2d(8, 4, kernel_size=(1, 1), stride=(1, 1))
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier_magnitude = nn.Linear(4, num_classes)
        self.classifier_complex = nn.Linear(4, num_classes)
        self.classifier_total = nn.Linear(4, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if isinstance(x, tuple):
            real_part, imag_part = x
        else:
            real_part = x
            imag_part = torch.zeros_like(x)

        complex_input = (real_part, imag_part)

        for layer in self.complex_layers:
            complex_input = layer(complex_input)

        real_final, imag_final = complex_input

        magnitude = complex_magnitude(complex_input)
        
        complex_component = torch.cat([real_final, imag_final], dim=1)

        Fm = self.magnitude_conv(magnitude)
        Fc = self.complex_conv(complex_component)

        Fm_pooled = self.global_avg_pool(Fm).squeeze(-1).squeeze(-1)
        Fc_pooled = self.global_avg_pool(Fc).squeeze(-1).squeeze(-1)

        Ft = torch.cat([Fc, Fm], dim=1)
        Fout = self.attention(Ft)
        T = self.total_conv(Fout)
        T_pooled = self.global_avg_pool(T).squeeze(-1).squeeze(-1)

        logits_magnitude = self.classifier_magnitude(Fm_pooled)
        logits_complex = self.classifier_complex(Fc_pooled)
        logits_total = self.classifier_total(T_pooled)

        return logits_magnitude, logits_complex, logits_total
    
    def get_anomaly_score(self, x, true_class):
        self.eval()
        with torch.no_grad():
            _, _, logits_total = self.forward(x)
            return F.cross_entropy(logits_total, true_class, reduction='none')
        
    def get_anomaly_score_uncertainty(self, x, true_class):
        self.eval()
        with torch.no_grad():
            logits_magnitude, logits_complex, logits_total = self.forward(x)
            
            probs = F.softmax(logits_total, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            return entropy

    def get_anomaly_score_max_prob(self, x, true_class):
        self.eval()
        with torch.no_grad():
            logits_magnitude, logits_complex, logits_total = self.forward(x)
            
            probs = F.softmax(logits_total, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            
            return 1.0 - max_probs
