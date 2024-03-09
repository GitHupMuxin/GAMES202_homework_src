#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
    
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.f);

            float id = frameInfo.m_id(x, y);
            if (id < 0)
                continue;
            Matrix4x4 invModel = Inverse(frameInfo.m_matrix[id]);
            Matrix4x4 preModel = m_preFrameInfo.m_matrix[id];
            Float3 WordPosition = frameInfo.m_position(x, y);

            Float3 screenPosition = invModel(WordPosition, Float3::EType::Point);
            screenPosition = preModel(screenPosition, Float3::EType::Point);
            screenPosition = preWorldToScreen(screenPosition, Float3::EType::Point);

            if (screenPosition.x < 0 || screenPosition.x >= width || screenPosition.y < 0 || screenPosition.y >= height)
                continue;
            float findId = m_preFrameInfo.m_id(screenPosition.x, screenPosition.y);
            if (id != findId)
                continue;
            m_valid(x, y) = true;
            m_misc(x, y) = m_accColor(screenPosition.x, screenPosition.y);
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
    int k = m_colorBoxK;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            float alpha = 1.0f;
            if (m_valid(x, y)) 
            {
                alpha = m_alpha;

                int xStar = std::max(0, x - kernelRadius);
                int xEnd = std::min(x + kernelRadius, width - 1);
                int yStar = std::max(0, x - kernelRadius);
                int yEnd = std::min(x + kernelRadius, height - 1);

                Float3 mu(0.0);
                Float3 sigma(0.0);

                for (int i = xStar; i <= xEnd; i++) {
                    for (int j = yStar; j <= yEnd; j++) {
                        mu += curFilteredColor(x, y);
                        //sigma += Sqr(curFilteredColor(i, j));
                        sigma += Sqr(curFilteredColor(i, j) - curFilteredColor(x, y));
                    }
                }

                int count = pow(2 * kernelRadius + 1, 2);
                mu /= (float)count;
                //sigma = SafeSqrt(sigma / (float)count - Sqr(mu));
                sigma = SafeSqrt(sigma / (float)count);
                color = Clamp(color, mu - sigma * k, mu + sigma * k);
            }
            // TODO: Exponential moving average
            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 32;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter

            int xStar   = std::max(0, x - kernelRadius);
            int xEnd    = std::min(x + kernelRadius, width - 1);
            int yStar   = std::max(0, y - kernelRadius);
            int yEnd    = std::min(y + kernelRadius, height - 1);

            float sigmaCoord    = this->m_sigmaCoord;
            float sigmaColor    = this->m_sigmaColor;
            float sigmaNormal   = this->m_sigmaNormal;
            float sigmaPlane    = this->m_sigmaPlane; 

            Float3 nowPosition  = frameInfo.m_position(x, y);
            Float3 nowColor     = frameInfo.m_beauty(x, y);
            Float3 nowNormal    = frameInfo.m_normal(x, y);

            Float3 result = Float3(0.0);
            float totalWeight = 0.0;

            for (int i = xStar; i <= xEnd; i++)
            {
                for (int j = yStar; j <= yEnd; j++)
                {
                    Float3 curPosition  = frameInfo.m_position(i, j);
                    Float3 curColor     = frameInfo.m_beauty(i, j);
                    Float3 curNormal    = frameInfo.m_normal(i, j);
                    
                    float sqrDistance   = SqrDistance(curPosition, nowPosition);
                    float sqrDColor     = SqrDistance(nowColor, curColor);
                    float sqrDNormal    = std::pow(SafeAcos(Dot(nowNormal, curNormal)), 2);
                    float sqrDPlane     = sqrDistance > 0 ? std::pow(Dot(nowNormal, Normalize(curPosition - nowPosition)), 2) : 0;

                    float weightOfPosition  = -sqrDistance / (2 * sigmaCoord * sigmaCoord);
                    float weightOfColor     = -sqrDColor / (2 * sigmaColor * sigmaColor);
                    float weightOfNormal    = -sqrDNormal / (2 * sigmaNormal * sigmaNormal);
                    float weightOfPlane     = -sqrDPlane / (2 * sigmaPlane * sigmaPlane);

                    float weight = std::exp(weightOfPosition + weightOfColor + weightOfNormal + weightOfPlane);
                    weight = std::max(weight, 0.000001f);
                    result += curColor * weight;
                    totalWeight += weight;
                }
            }
            filteredImage(x, y) = result / totalWeight;
        }
    }
    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
