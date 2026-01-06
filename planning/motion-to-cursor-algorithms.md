# Motion-to-Cursor Algorithms for Smartwatch IMU Data

Comprehensive research on algorithms for translating smartwatch IMU (Inertial Measurement Unit) data into cursor movement, covering sensor fusion, positioning modes, drift correction, user experience optimization, and gesture detection.

---

## Table of Contents

1. [Sensor Fusion](#1-sensor-fusion)
2. [Positioning Modes](#2-positioning-modes)
3. [Drift Correction](#3-drift-correction)
4. [User Experience](#4-user-experience)
5. [Gesture Detection](#5-gesture-detection)
6. [Integration Recommendations](#6-integration-recommendations)
7. [References](#7-references)

---

## 1. Sensor Fusion

Sensor fusion combines data from multiple IMU sensors (accelerometer, gyroscope, magnetometer) to obtain accurate orientation and motion estimates. Different algorithms offer trade-offs between computational complexity, accuracy, and real-time performance.

### 1.1 Complementary Filter

**Overview**: The simplest and most computationally efficient sensor fusion algorithm. It combines high-pass filtered gyroscope data with low-pass filtered accelerometer data.

**Core Concept**:
- Gyroscopes provide accurate short-term measurements but drift over time
- Accelerometers provide stable long-term reference but are noisy in short-term
- Complementary filter uses gyroscope for fast changes, accelerometer for long-term stability

**Algorithm**:
```
θ = α * θ_gyro + (1 - α) * θ_accel

where:
- θ: Estimated angle
- θ_gyro: Angle from gyroscope integration
- θ_accel: Angle from accelerometer
- α: Filter gain (typically 0.95-0.98)
```

**Pseudocode**:
```python
class ComplementaryFilter:
    def __init__(self, alpha=0.98, dt=0.01):
        self.alpha = alpha  # Gyroscope trust factor
        self.dt = dt        # Sample time (seconds)
        self.pitch = 0.0
        self.roll = 0.0

    def update(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        # Calculate angles from accelerometer (in degrees)
        accel_pitch = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * 180/math.pi
        accel_roll = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)) * 180/math.pi

        # Integrate gyroscope data (gyro is in deg/sec)
        self.pitch += gyro_x * self.dt
        self.roll += gyro_y * self.dt

        # Apply complementary filter
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * accel_pitch
        self.roll = self.alpha * self.roll + (1 - self.alpha) * accel_roll

        return self.pitch, self.roll
```

**Advantages**:
- Very low computational cost (109 scalar operations per update)
- Easy to implement and tune
- Works well for real-time applications on resource-constrained devices
- No matrix operations required

**Disadvantages**:
- Cannot estimate yaw without magnetometer
- Less accurate than Kalman filter for complex motion
- Fixed gain doesn't adapt to changing conditions

**Tuning Guidelines**:
- α = 0.95: More responsive but more susceptible to vibration
- α = 0.98: Good balance for wearable devices
- α = 0.99: Very stable but slow to respond
- Sample rate (dt): 10-20ms (50-100Hz) recommended for cursor control

---

### 1.2 Madgwick Filter

**Overview**: A computationally efficient orientation filter that uses gradient descent optimization to fuse accelerometer, gyroscope, and optionally magnetometer data using quaternions.

**Core Concept**:
- Uses quaternion representation to avoid gimbal lock
- Optimizes the orientation quaternion to minimize the difference between measured and predicted gravity/magnetic field
- Combines gyroscope integration with accelerometer/magnetometer correction

**Mathematical Foundation**:
```
Quaternion update: q̇ = 0.5 * q ⊗ [0, ωx, ωy, ωz] - β * ∇f

where:
- q: Orientation quaternion
- ω: Angular velocity from gyroscope
- β: Gain parameter (controls correction rate)
- ∇f: Gradient of error function
- ⊗: Quaternion multiplication
```

**Pseudocode**:
```python
class MadgwickFilter:
    def __init__(self, beta=0.1, sample_freq=100):
        self.beta = beta  # Gain parameter (0.033-0.1)
        self.sample_freq = sample_freq
        self.q = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]

    def update_imu(self, accel, gyro):
        """Update with accelerometer and gyroscope only (6DOF)"""
        ax, ay, az = accel
        gx, gy, gz = gyro

        # Normalize accelerometer measurement
        norm = math.sqrt(ax**2 + ay**2 + az**2)
        if norm == 0:
            return self.q
        ax, ay, az = ax/norm, ay/norm, az/norm

        # Extract quaternion components
        q0, q1, q2, q3 = self.q

        # Gradient descent algorithm corrective step
        s0 = -2*q2*(2*q1*q3 - 2*q0*q2 - ax) + 2*q1*(2*q0*q1 + 2*q2*q3 - ay)
        s1 = 2*q3*(2*q1*q3 - 2*q0*q2 - ax) + 2*q0*(2*q0*q1 + 2*q2*q3 - ay) - 4*q1*(1 - 2*q1**2 - 2*q2**2 - az)
        s2 = -2*q0*(2*q1*q3 - 2*q0*q2 - ax) + 2*q3*(2*q0*q1 + 2*q2*q3 - ay) - 4*q2*(1 - 2*q1**2 - 2*q2**2 - az)
        s3 = 2*q1*(2*q1*q3 - 2*q0*q2 - ax) + 2*q2*(2*q0*q1 + 2*q2*q3 - ay)

        # Normalize step magnitude
        norm = math.sqrt(s0**2 + s1**2 + s2**2 + s3**2)
        s0, s1, s2, s3 = s0/norm, s1/norm, s2/norm, s3/norm

        # Convert gyroscope to rad/s
        gx, gy, gz = gx * math.pi/180, gy * math.pi/180, gz * math.pi/180

        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-q1*gx - q2*gy - q3*gz) - self.beta * s0
        qDot2 = 0.5 * (q0*gx + q2*gz - q3*gy) - self.beta * s1
        qDot3 = 0.5 * (q0*gy - q1*gz + q3*gx) - self.beta * s2
        qDot4 = 0.5 * (q0*gz + q1*gy - q2*gx) - self.beta * s3

        # Integrate to yield quaternion
        dt = 1.0 / self.sample_freq
        q0 += qDot1 * dt
        q1 += qDot2 * dt
        q2 += qDot3 * dt
        q3 += qDot4 * dt

        # Normalize quaternion
        norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        self.q = [q0/norm, q1/norm, q2/norm, q3/norm]

        return self.q

    def get_euler_angles(self):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        q0, q1, q2, q3 = self.q

        # Roll (x-axis rotation)
        roll = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))

        # Pitch (y-axis rotation)
        pitch = math.asin(2*(q0*q2 - q3*q1))

        # Yaw (z-axis rotation) - not accurate without magnetometer
        yaw = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

        return roll, pitch, yaw
```

**Advantages**:
- Quaternion representation avoids gimbal lock
- More accurate than complementary filter
- Can include magnetometer for full 9DOF orientation
- Computationally efficient (109 arithmetic operations per update)
- Handles dynamic motion well

**Disadvantages**:
- More complex to implement than complementary filter
- Requires tuning of β parameter
- Quaternion math may be unfamiliar to developers

**Tuning Guidelines**:
- β = 0.033: Recommended for IMU (no magnetometer)
- β = 0.041: Recommended for MARG (with magnetometer)
- β = 0.1: More aggressive correction for high dynamics
- Lower β = smoother but slower convergence
- Higher β = faster convergence but more noise

**Implementation Resources**:
- [x-io Technologies Open Source Implementation](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/) (C, C#, MATLAB)
- [AHRS Python Library](https://ahrs.readthedocs.io/en/latest/filters/madgwick.html)
- [GitHub: bjohnsonfl/Madgwick_Filter](https://github.com/bjohnsonfl/Madgwick_Filter)

---

### 1.3 Mahony Filter

**Overview**: A complementary filter enhanced with proportional-integral (PI) control for gyroscope bias estimation and correction.

**Core Concept**:
- Extends complementary filter with explicit gyroscope bias correction
- Uses PI controller to minimize error between accelerometer and gyroscope estimates
- Estimates and removes gyroscope bias in real-time

**Mathematical Foundation**:
```
ω_corrected = ω_gyro - b + Kp * e + Ki * ∫e dt

where:
- ω_corrected: Corrected angular velocity
- ω_gyro: Raw gyroscope measurement
- b: Estimated gyroscope bias
- e: Error between accelerometer and gyroscope estimates
- Kp: Proportional gain (typically 1.0)
- Ki: Integral gain (typically 0.3)
```

**Pseudocode**:
```python
class MahonyFilter:
    def __init__(self, kp=1.0, ki=0.3, sample_freq=100):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.sample_freq = sample_freq
        self.q = [1.0, 0.0, 0.0, 0.0]  # Quaternion [w, x, y, z]
        self.integral_error = [0.0, 0.0, 0.0]  # Integral of gyro error

    def update(self, accel, gyro):
        ax, ay, az = accel
        gx, gy, gz = gyro

        # Normalize accelerometer
        norm = math.sqrt(ax**2 + ay**2 + az**2)
        if norm == 0:
            return self.q
        ax, ay, az = ax/norm, ay/norm, az/norm

        q0, q1, q2, q3 = self.q

        # Estimated direction of gravity (from quaternion)
        vx = 2*(q1*q3 - q0*q2)
        vy = 2*(q0*q1 + q2*q3)
        vz = q0**2 - q1**2 - q2**2 + q3**2

        # Error is cross product between estimated and measured gravity
        ex = (ay*vz - az*vy)
        ey = (az*vx - ax*vz)
        ez = (ax*vy - ay*vx)

        # Apply integral feedback
        if self.ki > 0:
            dt = 1.0 / self.sample_freq
            self.integral_error[0] += ex * dt
            self.integral_error[1] += ey * dt
            self.integral_error[2] += ez * dt

        # Apply proportional and integral feedback to gyroscope
        gx += self.kp * ex + self.ki * self.integral_error[0]
        gy += self.kp * ey + self.ki * self.integral_error[1]
        gz += self.kp * ez + self.ki * self.integral_error[2]

        # Convert gyroscope to rad/s
        gx, gy, gz = gx * math.pi/180, gy * math.pi/180, gz * math.pi/180

        # Integrate quaternion rate
        dt = 1.0 / self.sample_freq
        q0 += 0.5 * (-q1*gx - q2*gy - q3*gz) * dt
        q1 += 0.5 * (q0*gx + q2*gz - q3*gy) * dt
        q2 += 0.5 * (q0*gy - q1*gz + q3*gx) * dt
        q3 += 0.5 * (q0*gz + q1*gy - q2*gx) * dt

        # Normalize quaternion
        norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        self.q = [q0/norm, q1/norm, q2/norm, q3/norm]

        return self.q

    def get_euler_angles(self):
        """Convert quaternion to Euler angles"""
        q0, q1, q2, q3 = self.q
        roll = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = math.asin(2*(q0*q2 - q3*q1))
        yaw = math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        return roll, pitch, yaw
```

**Advantages**:
- Automatically estimates and corrects gyroscope bias
- Better accuracy than basic complementary filter
- Computationally efficient (comparable to Madgwick)
- Good for devices with gyroscope bias drift
- PI controller provides robust error correction

**Disadvantages**:
- Slightly more complex than complementary filter
- Requires tuning of Kp and Ki parameters
- Integral term can wind up if not properly bounded

**Tuning Guidelines**:
- Kp = 1.0, Ki = 0.3: Standard values from research
- Kp = 2.0, Ki = 0.0: Pure proportional (faster, more noise)
- Kp = 0.5, Ki = 0.1: More stable, slower convergence
- For cursor control: Start with Kp = 1.0, Ki = 0.3

**Comparison**: Research shows Mahony filter often provides best orientation estimation on quadcopters when parameters are optimized.

---

### 1.4 Kalman Filter

**Overview**: An optimal recursive state estimator that provides minimum mean-square error estimates by modeling system dynamics and measurement noise.

**Core Concept**:
- Prediction step: Predict state using system model
- Update step: Correct prediction using sensor measurements
- Dynamically adjusts trust between prediction and measurement based on noise characteristics
- Can estimate additional states (position, velocity, bias)

**Mathematical Foundation**:
```
Prediction:
x̂_k|k-1 = F * x̂_k-1|k-1 + B * u_k
P_k|k-1 = F * P_k-1|k-1 * F^T + Q

Update:
K_k = P_k|k-1 * H^T * (H * P_k|k-1 * H^T + R)^-1
x̂_k|k = x̂_k|k-1 + K_k * (z_k - H * x̂_k|k-1)
P_k|k = (I - K_k * H) * P_k|k-1

where:
- x̂: State estimate (orientation, angular velocity, bias)
- P: Error covariance matrix
- F: State transition matrix
- Q: Process noise covariance
- R: Measurement noise covariance
- H: Measurement matrix
- K: Kalman gain
```

**Pseudocode** (Simplified Linear Kalman Filter for orientation):
```python
import numpy as np

class KalmanFilter:
    def __init__(self, dt=0.01):
        self.dt = dt

        # State vector: [roll, pitch, roll_rate, pitch_rate]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (we measure angles directly from accel)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance (tunable)
        self.Q = np.eye(4) * 0.001

        # Measurement noise covariance (tunable)
        self.R = np.eye(2) * 0.1

        # Error covariance matrix
        self.P = np.eye(4)

    def predict(self, gyro_x, gyro_y):
        """Prediction step using gyroscope data"""
        # Update state transition with gyro measurements
        u = np.array([[gyro_x], [gyro_y]])

        # Predict state
        self.x = self.F @ self.x
        self.x[2:4] = u  # Update angular velocities with gyro

        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, accel_x, accel_y, accel_z):
        """Update step using accelerometer data"""
        # Calculate angles from accelerometer
        roll = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2))
        pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2))

        z = np.array([[roll], [pitch]])

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update error covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    def get_orientation(self):
        """Return current roll and pitch estimates"""
        return float(self.x[0]), float(self.x[1])

    def get_angular_velocity(self):
        """Return angular velocity estimates"""
        return float(self.x[2]), float(self.x[3])
```

**Advantages**:
- Optimal estimate under Gaussian noise assumptions
- Dynamically adapts to changing conditions
- Can estimate additional states (position, bias, velocity)
- Provides uncertainty estimates (covariance)
- Well-suited for cursor position tracking

**Disadvantages**:
- More computationally expensive (matrix operations)
- Requires careful tuning of Q and R matrices
- Complex implementation
- May be overkill for orientation-only tracking

**Tuning Guidelines**:
- **Process Noise (Q)**: Represents uncertainty in motion model
  - Larger Q: Trust measurements more (faster adaptation, more noise)
  - Smaller Q: Trust model more (smoother, slower adaptation)
  - Typical values: 0.0001 to 0.01 for orientation states

- **Measurement Noise (R)**: Represents sensor accuracy
  - Based on actual sensor characteristics
  - Accelerometer: R ≈ 0.1 to 1.0 (depends on vibration environment)
  - Gyroscope: R ≈ 0.001 to 0.01 (very accurate for short-term)

**Use Cases for Cursor Control**:
- Position estimation (double integration of accelerometer)
- Cursor velocity smoothing
- Predictive cursor movement
- Multi-sensor fusion (IMU + optical tracking)

**Research Application**: A study demonstrated cursor control using 4 IMUs on upper body with Kalman filter, achieving significant performance improvements by leveraging signal redundancy.

---

### 1.5 Filter Selection Guide

**For Smartwatch Cursor Control:**

| Filter | Computational Cost | Accuracy | Drift Resistance | Recommended Use Case |
|--------|-------------------|----------|------------------|---------------------|
| **Complementary** | Very Low | Good | Moderate | Quick prototyping, battery-constrained devices |
| **Madgwick** | Low | Excellent | Excellent | Primary recommendation for orientation tracking |
| **Mahony** | Low | Excellent | Excellent | Best when gyroscope bias is significant |
| **Kalman** | High | Excellent | Excellent | Position tracking, multi-sensor fusion |

**Implementation Recommendation**:
1. **Start with Madgwick filter** for orientation tracking
   - Best balance of accuracy and efficiency
   - Well-tested in wearable applications
   - Good drift resistance with quaternion representation

2. **Use Complementary filter** if:
   - Extreme battery constraints
   - Very simple implementation needed
   - Only 2-axis control (roll/pitch) required

3. **Use Mahony filter** if:
   - Gyroscope bias drift is problematic
   - Need explicit bias estimation
   - Similar performance to Madgwick with different tuning characteristics

4. **Use Kalman filter** if:
   - Tracking cursor position (not just orientation)
   - Fusing with additional sensors (optical, GPS)
   - Need uncertainty estimates
   - Computational resources available

---

## 2. Positioning Modes

Different positioning modes translate IMU orientation/motion into cursor movement. Each mode has distinct characteristics suitable for different use cases.

### 2.1 Relative Positioning (Tilt-to-Move)

**Overview**: Hand tilt controls cursor velocity, similar to an analog joystick. The cursor moves continuously while the hand is tilted, with tilt angle determining speed and direction.

**Core Concept**:
```
cursor_velocity = orientation * sensitivity_factor
cursor_position += cursor_velocity * dt
```

**Algorithm**:
```python
class RelativePositioning:
    def __init__(self, sensitivity=2.0, max_speed=1000):
        self.sensitivity = sensitivity  # pixels/degree
        self.max_speed = max_speed      # pixels/second
        self.cursor_x = 0
        self.cursor_y = 0

    def update(self, roll, pitch, dt):
        """
        roll: rotation around x-axis (degrees)
        pitch: rotation around y-axis (degrees)
        dt: time step (seconds)
        """
        # Calculate velocity from tilt angles
        # Pitch controls Y movement (forward/back)
        # Roll controls X movement (left/right)
        velocity_x = roll * self.sensitivity
        velocity_y = -pitch * self.sensitivity  # Negative for natural mapping

        # Clamp velocities to max speed
        velocity_x = max(-self.max_speed, min(self.max_speed, velocity_x))
        velocity_y = max(-self.max_speed, min(self.max_speed, velocity_y))

        # Update cursor position
        self.cursor_x += velocity_x * dt
        self.cursor_y += velocity_y * dt

        # Clamp to screen bounds
        self.cursor_x = max(0, min(screen_width, self.cursor_x))
        self.cursor_y = max(0, min(screen_height, self.cursor_y))

        return int(self.cursor_x), int(self.cursor_y)
```

**Enhanced Version with Dead Zone and Non-Linear Response**:
```python
class EnhancedRelativePositioning:
    def __init__(self, sensitivity=2.0, max_speed=1200,
                 dead_zone=2.0, acceleration_curve=2.0):
        self.sensitivity = sensitivity
        self.max_speed = max_speed
        self.dead_zone = dead_zone        # degrees
        self.acceleration_curve = acceleration_curve  # 1.0 = linear, 2.0 = quadratic
        self.cursor_x = 0.0
        self.cursor_y = 0.0

    def apply_dead_zone(self, value, threshold):
        """Apply dead zone with smooth transition"""
        if abs(value) < threshold:
            return 0.0
        # Linear scaling outside dead zone
        sign = 1 if value > 0 else -1
        magnitude = abs(value) - threshold
        max_value = 90 - threshold  # Assuming max tilt of 90 degrees
        return sign * (magnitude / max_value) * 90

    def apply_curve(self, value, curve):
        """Apply acceleration curve to velocity"""
        if value == 0:
            return 0
        sign = 1 if value > 0 else -1
        normalized = abs(value) / 90.0  # Normalize to 0-1
        curved = normalized ** curve
        return sign * curved * 90.0

    def update(self, roll, pitch, dt):
        # Apply dead zone
        roll = self.apply_dead_zone(roll, self.dead_zone)
        pitch = self.apply_dead_zone(pitch, self.dead_zone)

        # Apply acceleration curve
        roll = self.apply_curve(roll, self.acceleration_curve)
        pitch = self.apply_curve(pitch, self.acceleration_curve)

        # Calculate velocities
        velocity_x = roll * self.sensitivity
        velocity_y = -pitch * self.sensitivity

        # Clamp to max speed
        velocity_x = np.clip(velocity_x, -self.max_speed, self.max_speed)
        velocity_y = np.clip(velocity_y, -self.max_speed, self.max_speed)

        # Update position
        self.cursor_x += velocity_x * dt
        self.cursor_y += velocity_y * dt

        # Screen bounds
        self.cursor_x = np.clip(self.cursor_x, 0, screen_width)
        self.cursor_y = np.clip(self.cursor_y, 0, screen_height)

        return int(self.cursor_x), int(self.cursor_y)
```

**Advantages**:
- Natural feeling similar to joystick/gamepad
- No absolute position tracking needed (no drift accumulation)
- Works from any hand position
- Easy to implement
- Low fatigue (can rest hand in neutral position)

**Disadvantages**:
- Cannot instantly jump to screen locations
- Less precise for pointing tasks
- Requires continuous motion for movement
- Speed-accuracy trade-off

**Best Use Cases**:
- Gaming
- Media control
- Scrolling interfaces
- Situations where user is moving (walking)
- When fatigue reduction is priority

**Tuning Parameters**:
- **Sensitivity**: 1.0-5.0 pixels/degree (start with 2.0)
- **Max Speed**: 800-1500 pixels/second (1000 typical)
- **Dead Zone**: 1.5-3.0 degrees (2.0 recommended)
- **Acceleration Curve**: 1.5-2.5 (2.0 for quadratic feel)

---

### 2.2 Absolute Positioning (Point-at-Screen)

**Overview**: Hand position/orientation maps directly to screen coordinates. Pointing the hand at a screen location moves cursor there directly.

**Core Concept**:
```
cursor_position = screen_map(hand_orientation)
```

**Algorithm**:
```python
class AbsolutePositioning:
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Calibration: map orientation ranges to screen space
        # These would be determined during calibration
        self.pitch_min = -30  # degrees (top of screen)
        self.pitch_max = 30   # degrees (bottom of screen)
        self.roll_min = -40   # degrees (left of screen)
        self.roll_max = 40    # degrees (right of screen)

        # Reference orientation (calibrated center position)
        self.roll_center = 0
        self.pitch_center = 0

    def calibrate(self, corners):
        """
        Calibrate screen mapping from 4 corner orientations
        corners: list of (roll, pitch) for [top-left, top-right, bottom-left, bottom-right]
        """
        tl, tr, bl, br = corners

        self.roll_min = (tl[0] + bl[0]) / 2
        self.roll_max = (tr[0] + br[0]) / 2
        self.pitch_min = (tl[1] + tr[1]) / 2
        self.pitch_max = (bl[1] + br[1]) / 2

        self.roll_center = (self.roll_min + self.roll_max) / 2
        self.pitch_center = (self.pitch_min + self.pitch_max) / 2

    def update(self, roll, pitch):
        """Map orientation to screen coordinates"""
        # Normalize to 0-1 range
        roll_normalized = (roll - self.roll_min) / (self.roll_max - self.roll_min)
        pitch_normalized = (pitch - self.pitch_min) / (self.pitch_max - self.pitch_min)

        # Clamp to valid range
        roll_normalized = max(0, min(1, roll_normalized))
        pitch_normalized = max(0, min(1, pitch_normalized))

        # Map to screen coordinates
        cursor_x = roll_normalized * self.screen_width
        cursor_y = pitch_normalized * self.screen_height

        return int(cursor_x), int(cursor_y)
```

**Enhanced Version with Smoothing and Edge Zones**:
```python
class EnhancedAbsolutePositioning:
    def __init__(self, screen_width=1920, screen_height=1080,
                 smoothing_factor=0.3, edge_zone=50):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing_factor = smoothing_factor  # 0=no smoothing, 1=maximum
        self.edge_zone = edge_zone  # pixels from edge for snap

        # Calibration ranges
        self.pitch_min = -30
        self.pitch_max = 30
        self.roll_min = -40
        self.roll_max = 40

        # Smoothed cursor position
        self.cursor_x = screen_width / 2
        self.cursor_y = screen_height / 2

        # History for velocity calculation
        self.prev_raw_x = self.cursor_x
        self.prev_raw_y = self.cursor_y

    def apply_smoothing(self, current, target, factor):
        """Exponential moving average smoothing"""
        return current + (target - current) * (1 - factor)

    def apply_edge_snap(self, position, max_position, edge_zone):
        """Snap to edges within zone"""
        if position < edge_zone:
            return 0
        elif position > max_position - edge_zone:
            return max_position
        return position

    def update(self, roll, pitch):
        # Map orientation to raw screen coordinates
        roll_norm = (roll - self.roll_min) / (self.roll_max - self.roll_min)
        pitch_norm = (pitch - self.pitch_min) / (self.pitch_max - self.pitch_min)

        roll_norm = np.clip(roll_norm, 0, 1)
        pitch_norm = np.clip(pitch_norm, 0, 1)

        raw_x = roll_norm * self.screen_width
        raw_y = pitch_norm * self.screen_height

        # Apply edge snapping
        raw_x = self.apply_edge_snap(raw_x, self.screen_width, self.edge_zone)
        raw_y = self.apply_edge_snap(raw_y, self.screen_height, self.edge_zone)

        # Apply smoothing
        self.cursor_x = self.apply_smoothing(self.cursor_x, raw_x,
                                             self.smoothing_factor)
        self.cursor_y = self.apply_smoothing(self.cursor_y, raw_y,
                                             self.smoothing_factor)

        # Update history
        self.prev_raw_x = raw_x
        self.prev_raw_y = raw_y

        return int(self.cursor_x), int(self.cursor_y)

    def get_velocity(self):
        """Calculate current velocity (for acceleration filtering)"""
        velocity_x = self.cursor_x - self.prev_raw_x
        velocity_y = self.cursor_y - self.prev_raw_y
        return math.sqrt(velocity_x**2 + velocity_y**2)
```

**Advantages**:
- Direct pointing interface (intuitive)
- Fast cursor positioning
- Precise control once calibrated
- No continuous motion needed
- Good for targeting tasks

**Disadvantages**:
- Requires calibration for each user/position
- Arm fatigue from holding arm extended
- Drift causes cursor drift
- Sensitive to body/device movement
- Limited workspace (arm range of motion)

**Best Use Cases**:
- Presentations
- Short interaction sessions
- Precise pointing tasks
- When user is stationary
- Large screen interactions (TV, projector)

**Tuning Parameters**:
- **Smoothing Factor**: 0.2-0.5 (0.3 balanced)
- **Edge Zone**: 30-100 pixels (50 typical)
- **Calibration Range**: ±30° pitch, ±40° roll (adjustable per user)

**Calibration Procedure**:
```python
def calibration_routine():
    """Interactive calibration for absolute positioning"""
    print("Point at TOP-LEFT corner and tap")
    top_left = wait_for_tap_and_record_orientation()

    print("Point at TOP-RIGHT corner and tap")
    top_right = wait_for_tap_and_record_orientation()

    print("Point at BOTTOM-LEFT corner and tap")
    bottom_left = wait_for_tap_and_record_orientation()

    print("Point at BOTTOM-RIGHT corner and tap")
    bottom_right = wait_for_tap_and_record_orientation()

    corners = [top_left, top_right, bottom_left, bottom_right]

    absolute_mode = AbsolutePositioning()
    absolute_mode.calibrate(corners)

    return absolute_mode
```

---

### 2.3 Hybrid Mode

**Overview**: Combines relative and absolute positioning, switching between modes based on context or using elements of both simultaneously.

**Approach 1: Mode Switching**
```python
class HybridModeSwitch:
    def __init__(self):
        self.relative_mode = EnhancedRelativePositioning()
        self.absolute_mode = EnhancedAbsolutePositioning()
        self.current_mode = 'relative'
        self.mode_switch_gesture_detected = False

    def switch_mode(self):
        """Toggle between modes"""
        if self.current_mode == 'relative':
            self.current_mode = 'absolute'
            # Sync absolute mode to current cursor position
            self.absolute_mode.cursor_x = self.relative_mode.cursor_x
            self.absolute_mode.cursor_y = self.relative_mode.cursor_y
        else:
            self.current_mode = 'relative'
            self.relative_mode.cursor_x = self.absolute_mode.cursor_x
            self.relative_mode.cursor_y = self.absolute_mode.cursor_y

    def update(self, roll, pitch, dt):
        """Update cursor based on current mode"""
        # Check for mode switch gesture (e.g., palm-up gesture)
        if self.mode_switch_gesture_detected:
            self.switch_mode()
            self.mode_switch_gesture_detected = False

        if self.current_mode == 'relative':
            return self.relative_mode.update(roll, pitch, dt)
        else:
            return self.absolute_mode.update(roll, pitch)
```

**Approach 2: Velocity-Based Blending**
```python
class HybridVelocityBlending:
    def __init__(self, velocity_threshold=50):
        self.relative_mode = EnhancedRelativePositioning()
        self.absolute_mode = EnhancedAbsolutePositioning()
        self.velocity_threshold = velocity_threshold  # pixels/second
        self.cursor_x = 0
        self.cursor_y = 0

    def update(self, roll, pitch, dt, angular_velocity):
        """Blend modes based on motion speed"""
        # Get positions from both modes
        rel_x, rel_y = self.relative_mode.update(roll, pitch, dt)
        abs_x, abs_y = self.absolute_mode.update(roll, pitch)

        # Calculate blend factor based on angular velocity
        # Fast motion: use relative (more responsive)
        # Slow motion: use absolute (more precise)
        velocity_magnitude = math.sqrt(angular_velocity[0]**2 +
                                      angular_velocity[1]**2)

        # Blend factor: 0 = all absolute, 1 = all relative
        blend = min(1.0, velocity_magnitude / self.velocity_threshold)

        # Blend positions
        self.cursor_x = abs_x * (1 - blend) + rel_x * blend
        self.cursor_y = abs_y * (1 - blend) + rel_y * blend

        return int(self.cursor_x), int(self.cursor_y)
```

**Approach 3: Coarse-Fine Control**
```python
class HybridCoarseFine:
    """
    Absolute positioning for coarse movement,
    relative fine-tuning when near target
    """
    def __init__(self, fine_radius=100):
        self.absolute_mode = EnhancedAbsolutePositioning()
        self.relative_mode = EnhancedRelativePositioning(
            sensitivity=0.5,  # Lower sensitivity for fine control
            max_speed=200
        )
        self.fine_radius = fine_radius  # pixels
        self.in_fine_mode = False
        self.target_x = None
        self.target_y = None

    def update(self, roll, pitch, dt):
        # Always get absolute position as reference
        abs_x, abs_y = self.absolute_mode.update(roll, pitch)

        # Check if we have a target and are close to it
        if self.target_x is not None:
            distance = math.sqrt((abs_x - self.target_x)**2 +
                               (abs_y - self.target_y)**2)

            if distance < self.fine_radius:
                # Switch to fine mode
                if not self.in_fine_mode:
                    self.in_fine_mode = True
                    # Sync relative mode to current position
                    self.relative_mode.cursor_x = abs_x
                    self.relative_mode.cursor_y = abs_y

                # Use relative for fine control
                return self.relative_mode.update(roll, pitch, dt)
            else:
                self.in_fine_mode = False

        # Use absolute for coarse movement
        return abs_x, abs_y

    def set_target(self, x, y):
        """Set target for fine control (e.g., from voice command)"""
        self.target_x = x
        self.target_y = y
```

**Advantages**:
- Combines strengths of both modes
- Adapts to user intent and motion characteristics
- Provides fast coarse movement and precise fine control
- Reduces fatigue by using optimal mode for each situation

**Disadvantages**:
- More complex implementation
- Requires careful tuning of transition parameters
- May feel inconsistent if transitions are abrupt
- Higher computational cost

**Best Use Cases**:
- Professional applications requiring both speed and precision
- Users with varying mobility levels
- Multi-monitor setups
- Long usage sessions (reduces fatigue)

---

## 3. Drift Correction

Drift is the accumulation of errors over time in orientation and position estimates, caused by sensor noise, bias, and integration errors.

### 3.1 Understanding Drift Sources

**Gyroscope Drift**:
- Bias offset error: Non-zero output when stationary
- Temperature drift: Bias changes with temperature
- Integration drift: Errors accumulate when integrating angular velocity

**Accelerometer Drift**:
- Double integration: Position requires integrating acceleration twice
- Small errors grow quadratically over time
- Vibration and movement cause large errors

**Magnetometer Issues**:
- Magnetic interference from electronics
- Hard iron and soft iron distortions
- Not useful indoors (too much interference)

### 3.2 Sensor Fusion for Drift Mitigation

**Primary Strategy**: Use sensor fusion (Madgwick, Mahony, Kalman) as discussed in Section 1. These algorithms inherently reduce drift by:
- Correcting gyroscope with accelerometer/magnetometer
- Estimating and removing sensor bias
- Providing long-term stability

### 3.3 Zero-Velocity Update (ZUPT)

**Overview**: Detect when device is stationary and reset accumulated errors.

**Algorithm**:
```python
class ZeroVelocityUpdate:
    def __init__(self, accel_threshold=0.05, gyro_threshold=0.5,
                 window_size=10):
        self.accel_threshold = accel_threshold  # g
        self.gyro_threshold = gyro_threshold    # deg/s
        self.window_size = window_size
        self.accel_history = []
        self.gyro_history = []
        self.is_stationary = False

    def is_device_stationary(self, accel, gyro):
        """Detect if device is stationary"""
        # Add to history
        self.accel_history.append(accel)
        self.gyro_history.append(gyro)

        # Keep only recent samples
        if len(self.accel_history) > self.window_size:
            self.accel_history.pop(0)
            self.gyro_history.pop(0)

        if len(self.accel_history) < self.window_size:
            return False

        # Calculate variance of recent samples
        accel_array = np.array(self.accel_history)
        gyro_array = np.array(self.gyro_history)

        accel_var = np.var(np.linalg.norm(accel_array, axis=1))
        gyro_var = np.var(np.linalg.norm(gyro_array, axis=1))

        # Check if variance is below threshold
        stationary = (accel_var < self.accel_threshold and
                     gyro_var < self.gyro_threshold)

        self.is_stationary = stationary
        return stationary

    def apply_zupt(self, velocity_estimate, position_estimate):
        """Reset velocity when stationary"""
        if self.is_stationary:
            velocity_estimate *= 0  # Zero out velocity
        return velocity_estimate, position_estimate
```

**Usage Example**:
```python
zupt = ZeroVelocityUpdate()
velocity = np.array([0.0, 0.0, 0.0])

while True:
    accel, gyro = read_imu()

    # Check if stationary
    if zupt.is_device_stationary(accel, gyro):
        # Apply zero-velocity update
        velocity, position = zupt.apply_zupt(velocity, position)
        print("Stationary detected - resetting velocity")
    else:
        # Normal integration
        velocity += accel * dt
        position += velocity * dt
```

**Advantages**:
- Prevents unbounded drift growth
- No external reference needed
- Works well for wearables (frequent rest periods)

**Limitations**:
- Only works during stationary periods
- Can't correct drift during continuous motion

---

### 3.4 Periodic Recalibration

**Overview**: Periodically reset or recalibrate the IMU to a known reference state.

**Strategy 1: Manual Recalibration**
```python
class ManualRecalibration:
    def __init__(self):
        self.reference_orientation = None
        self.bias_samples = []
        self.calibration_samples = 100

    def start_calibration(self):
        """Begin calibration process"""
        self.bias_samples = []
        print("Hold device steady...")

    def collect_sample(self, accel, gyro):
        """Collect samples during calibration"""
        self.bias_samples.append((accel, gyro))

        if len(self.bias_samples) >= self.calibration_samples:
            return self.complete_calibration()

        return False  # Not done yet

    def complete_calibration(self):
        """Calculate bias from collected samples"""
        accels = np.array([s[0] for s in self.bias_samples])
        gyros = np.array([s[1] for s in self.bias_samples])

        self.accel_bias = np.mean(accels, axis=0)
        self.gyro_bias = np.mean(gyros, axis=0)

        # Accelerometer should read [0, 0, 1g] when flat
        # Subtract actual reading to get bias
        self.accel_bias[2] -= 1.0  # 1g gravity

        print(f"Calibration complete:")
        print(f"Accel bias: {self.accel_bias}")
        print(f"Gyro bias: {self.gyro_bias}")

        return True

    def apply_calibration(self, accel, gyro):
        """Apply calibration to sensor readings"""
        if hasattr(self, 'accel_bias'):
            accel = accel - self.accel_bias
            gyro = gyro - self.gyro_bias
        return accel, gyro
```

**Strategy 2: Automatic Drift Detection and Correction**
```python
class AutomaticDriftCorrection:
    def __init__(self, drift_threshold=5.0, correction_rate=0.1):
        self.drift_threshold = drift_threshold  # degrees
        self.correction_rate = correction_rate
        self.expected_gravity = np.array([0, 0, 1.0])  # Expected gravity vector
        self.orientation_offset = 0.0

    def detect_drift(self, orientation, accel):
        """
        Detect drift by comparing expected vs actual gravity direction
        """
        # Rotate expected gravity by current orientation
        # (simplified - actual implementation uses quaternion rotation)
        roll, pitch, yaw = orientation

        # Expected gravity in sensor frame based on orientation
        expected_accel = self.rotate_vector(self.expected_gravity,
                                           roll, pitch, yaw)

        # Normalize accelerometer reading
        accel_normalized = accel / np.linalg.norm(accel)

        # Calculate error
        error = np.linalg.norm(expected_accel - accel_normalized)
        error_degrees = math.degrees(math.asin(min(1.0, error)))

        return error_degrees

    def apply_correction(self, orientation, error_degrees):
        """Gradually correct orientation drift"""
        if error_degrees > self.drift_threshold:
            # Apply correction proportional to error
            correction = error_degrees * self.correction_rate
            # Apply to orientation (simplified)
            roll, pitch, yaw = orientation
            # Actual implementation would use quaternion slerp
            return (roll, pitch, yaw)  # with correction applied
        return orientation

    def rotate_vector(self, v, roll, pitch, yaw):
        """Rotate vector by Euler angles (simplified)"""
        # Full implementation would use rotation matrices
        # or quaternion math
        return v  # placeholder
```

**Strategy 3: Reference Position Reset**
```python
class ReferencePositionReset:
    """Reset cursor position to known reference on demand"""

    def __init__(self, reset_gesture_detected=False):
        self.home_position = (screen_width // 2, screen_height // 2)
        self.reset_gesture_detected = reset_gesture_detected

    def check_reset_gesture(self, orientation, gesture_state):
        """Detect reset gesture (e.g., palm-up + tap)"""
        # Example: palm facing up with tap
        roll, pitch, yaw = orientation

        if pitch < -45 and gesture_state == 'tap':  # Palm up
            return True
        return False

    def reset_to_home(self, cursor_controller):
        """Reset cursor to home position"""
        cursor_controller.cursor_x = self.home_position[0]
        cursor_controller.cursor_y = self.home_position[1]

        # Reset orientation reference
        cursor_controller.reference_orientation = cursor_controller.current_orientation

        print("Cursor reset to home position")
```

---

### 3.5 Complementary Sensor Correction

**Strategy: Use External References**

**Magnetometer Correction** (for absolute heading):
```python
class MagnetometerDriftCorrection:
    def __init__(self, mag_weight=0.01):
        self.mag_weight = mag_weight
        self.magnetic_north = None
        self.yaw_offset = 0.0

    def calibrate_magnetometer(self, samples):
        """Calibrate magnetometer for hard/soft iron distortion"""
        # Collect samples while rotating device
        # Fit ellipsoid to samples
        # Calculate correction matrix
        # (Simplified - actual implementation is complex)
        pass

    def correct_yaw_drift(self, yaw_gyro, magnetometer):
        """Use magnetometer to correct yaw drift"""
        # Calculate heading from magnetometer
        mag_x, mag_y, mag_z = magnetometer
        yaw_mag = math.atan2(mag_y, mag_x)

        # Blend with gyroscope-based yaw
        # Higher weight on magnetometer for long-term correction
        yaw_corrected = (1 - self.mag_weight) * yaw_gyro + \
                       self.mag_weight * yaw_mag

        return yaw_corrected
```

**Note**: Magnetometers are generally not reliable for indoor cursor control due to electromagnetic interference. They work better in outdoor/mobile robotics applications.

---

### 3.6 Adaptive Drift Correction

**Dynamic Adjustment Based on Usage Patterns**:
```python
class AdaptiveDriftCorrection:
    def __init__(self):
        self.drift_history = []
        self.correction_strength = 0.1
        self.usage_time = 0
        self.last_correction_time = 0

    def update(self, dt, orientation_error):
        """Adaptively adjust correction based on drift history"""
        self.usage_time += dt
        self.drift_history.append(orientation_error)

        # Keep last 1000 samples
        if len(self.drift_history) > 1000:
            self.drift_history.pop(0)

        # Calculate drift rate
        if len(self.drift_history) > 100:
            recent_drift = np.mean(self.drift_history[-100:])
            overall_drift = np.mean(self.drift_history)

            # If drift is accelerating, increase correction
            if recent_drift > overall_drift * 1.5:
                self.correction_strength = min(0.5,
                                              self.correction_strength * 1.1)
            else:
                self.correction_strength = max(0.05,
                                              self.correction_strength * 0.99)

        # Periodic recalibration prompt
        if self.usage_time - self.last_correction_time > 300:  # 5 minutes
            self.suggest_recalibration()
            self.last_correction_time = self.usage_time

    def suggest_recalibration(self):
        """Prompt user for recalibration"""
        avg_drift = np.mean(self.drift_history[-1000:])
        if avg_drift > 5.0:  # degrees
            print("Drift detected. Please recalibrate by holding device steady.")
```

---

### 3.7 Drift Correction Best Practices

**For Smartwatch Cursor Control:**

1. **Primary Defense**: Use Madgwick or Mahony filter
   - Inherent drift correction through accelerometer fusion
   - Gyroscope bias estimation (Mahony)

2. **Zero-Velocity Updates**: Implement ZUPT
   - Detect stationary periods (hand at rest)
   - Reset velocity accumulation
   - Especially important for position tracking

3. **Periodic Recalibration**:
   - Manual: Prompt every 5-10 minutes of active use
   - Automatic: Reset reference when palm-up gesture detected
   - Quick recalibration: 2-3 seconds of holding steady

4. **Reference Position Resets**:
   - Gesture to reset cursor to screen center
   - Voice command: "center cursor"
   - Automatic reset when returning from sleep

5. **Hybrid Positioning**:
   - Use relative positioning (no position drift)
   - Fall back to absolute only when needed
   - Regularly sync modes

6. **Temperature Compensation**:
   - If device has temperature sensor
   - Apply temperature-based bias correction
   - Particularly important for long sessions

7. **User Feedback**:
   - Visual indicator when drift detected
   - Haptic feedback during recalibration
   - Training users to recognize drift

**Implementation Priority**:
1. Madgwick/Mahony filter (must have)
2. ZUPT for stationary detection (high priority)
3. Manual recalibration gesture (medium priority)
4. Adaptive correction (nice to have)

---

## 4. User Experience

UX optimizations make cursor control feel natural, responsive, and precise while reducing fatigue and unintended inputs.

### 4.1 Dead Zones

**Purpose**: Ignore small unintentional movements to prevent jitter and provide stable rest position.

**Types of Dead Zones**:

**1. Radial Dead Zone** (recommended for 2D control):
```python
class RadialDeadZone:
    def __init__(self, dead_zone_radius=2.0):
        self.radius = dead_zone_radius  # degrees or normalized units

    def apply(self, x, y):
        """Apply radial dead zone with smooth transition"""
        magnitude = math.sqrt(x**2 + y**2)

        if magnitude < self.radius:
            # Inside dead zone - return zero
            return 0.0, 0.0

        # Outside dead zone - scale to remove dead zone offset
        # This creates smooth transition at boundary
        scale = (magnitude - self.radius) / magnitude
        return x * scale, y * scale
```

**2. Axial Dead Zone** (for independent X/Y control):
```python
class AxialDeadZone:
    def __init__(self, dead_zone_x=2.0, dead_zone_y=2.0):
        self.dead_zone_x = dead_zone_x
        self.dead_zone_y = dead_zone_y

    def apply(self, x, y):
        """Apply independent dead zones to each axis"""
        # Apply to X axis
        if abs(x) < self.dead_zone_x:
            x = 0
        else:
            sign = 1 if x > 0 else -1
            x = sign * (abs(x) - self.dead_zone_x)

        # Apply to Y axis
        if abs(y) < self.dead_zone_y:
            y = 0
        else:
            sign = 1 if y > 0 else -1
            y = sign * (abs(y) - self.dead_zone_y)

        return x, y
```

**3. Adaptive Dead Zone** (adjusts based on motion):
```python
class AdaptiveDeadZone:
    def __init__(self, min_radius=1.5, max_radius=4.0):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.current_radius = min_radius
        self.motion_history = []

    def update_radius(self, motion_variance):
        """Adjust dead zone based on recent motion patterns"""
        self.motion_history.append(motion_variance)
        if len(self.motion_history) > 50:
            self.motion_history.pop(0)

        avg_variance = np.mean(self.motion_history)

        # High variance (shaky hand) -> larger dead zone
        # Low variance (steady hand) -> smaller dead zone
        normalized_variance = min(1.0, avg_variance / 0.5)
        self.current_radius = self.min_radius + \
                             (self.max_radius - self.min_radius) * normalized_variance

    def apply(self, x, y):
        magnitude = math.sqrt(x**2 + y**2)

        if magnitude < self.current_radius:
            return 0.0, 0.0

        scale = (magnitude - self.current_radius) / magnitude
        return x * scale, y * scale
```

**Tuning Guidelines**:
- Small dead zone (1-2°): More responsive, may be jittery
- Medium dead zone (2-3°): Good balance for most users
- Large dead zone (3-5°): Very stable, may feel unresponsive
- Adaptive: Start with 2° minimum, 4° maximum

---

### 4.2 Sensitivity Curves

**Purpose**: Map input motion to cursor movement in non-linear ways for better control.

**1. Linear Sensitivity** (baseline):
```python
class LinearSensitivity:
    def __init__(self, sensitivity=2.0):
        self.sensitivity = sensitivity  # pixels per degree

    def apply(self, input_value):
        return input_value * self.sensitivity
```

**2. Exponential Curve** (slow for small movements, fast for large):
```python
class ExponentialSensitivity:
    def __init__(self, base_sensitivity=2.0, exponent=2.0):
        self.base_sensitivity = base_sensitivity
        self.exponent = exponent

    def apply(self, input_value):
        """Apply exponential curve"""
        sign = 1 if input_value >= 0 else -1
        magnitude = abs(input_value)

        # Normalize to 0-1 range (assuming max input is 90 degrees)
        normalized = min(1.0, magnitude / 90.0)

        # Apply exponential curve
        curved = normalized ** self.exponent

        # Scale back and apply sensitivity
        output = sign * curved * 90.0 * self.base_sensitivity

        return output
```

**3. S-Curve** (slow at edges, fast in middle):
```python
class SCurveSensitivity:
    def __init__(self, base_sensitivity=2.0, steepness=4.0):
        self.base_sensitivity = base_sensitivity
        self.steepness = steepness

    def apply(self, input_value):
        """Apply S-curve (sigmoid) transformation"""
        sign = 1 if input_value >= 0 else -1
        magnitude = abs(input_value)

        # Normalize to 0-1
        normalized = min(1.0, magnitude / 90.0)

        # Apply sigmoid curve
        # f(x) = 1 / (1 + e^(-k*(x-0.5)))
        # Shifted and scaled to go from 0 to 1
        x = (normalized - 0.5) * self.steepness
        curved = 1.0 / (1.0 + math.exp(-x))

        # Scale back
        output = sign * curved * 90.0 * self.base_sensitivity

        return output
```

**4. Power Curve with Adjustable Knee** (industry standard for gamepads):
```python
class PowerCurveSensitivity:
    def __init__(self, base_sensitivity=2.0, power=1.5, knee=0.5):
        self.base_sensitivity = base_sensitivity
        self.power = power
        self.knee = knee  # Inflection point (0-1)

    def apply(self, input_value):
        """
        Dual-slope power curve with adjustable knee point
        Allows fine control below knee, fast movement above
        """
        sign = 1 if input_value >= 0 else -1
        magnitude = abs(input_value)
        normalized = min(1.0, magnitude / 90.0)

        if normalized < self.knee:
            # Below knee: gentler curve
            section_progress = normalized / self.knee
            curved = (section_progress ** (self.power * 1.5)) * self.knee
        else:
            # Above knee: steeper curve
            section_progress = (normalized - self.knee) / (1.0 - self.knee)
            curved = self.knee + (section_progress ** (self.power * 0.5)) * (1.0 - self.knee)

        output = sign * curved * 90.0 * self.base_sensitivity
        return output
```

**Visual Comparison**:
```
Input (degrees): 0    22.5    45    67.5    90
─────────────────────────────────────────────
Linear (p=1.0):  0     25%    50%    75%   100%
Quadratic (p=2): 0      6%    25%    56%   100%
S-Curve:         0     12%    50%    88%   100%
Power (knee=.5): 0      3%    50%    81%   100%
```

**Recommendations**:
- **Beginners**: Linear or slight exponential (power=1.5)
- **General Use**: Power curve with knee=0.5, power=2.0
- **Precision Work**: S-curve for balanced control
- **Gaming**: Exponential with exponent=2.5

---

### 4.3 Acceleration/Deceleration Curves

**Purpose**: Smooth cursor starts and stops to prevent jarring movements.

**Basic Acceleration Limiting**:
```python
class AccelerationLimiter:
    def __init__(self, max_acceleration=5000):  # pixels/second²
        self.max_acceleration = max_acceleration
        self.current_velocity = np.array([0.0, 0.0])

    def apply(self, target_velocity, dt):
        """Limit rate of velocity change"""
        velocity_delta = target_velocity - self.current_velocity

        # Calculate required acceleration
        required_accel = velocity_delta / dt
        accel_magnitude = np.linalg.norm(required_accel)

        if accel_magnitude > self.max_acceleration:
            # Limit acceleration
            accel_limited = (required_accel / accel_magnitude) * \
                           self.max_acceleration
            self.current_velocity += accel_limited * dt
        else:
            # Can achieve target velocity
            self.current_velocity = target_velocity

        return self.current_velocity
```

**Smooth Start/Stop with Ease Curves**:
```python
class EaseCurve:
    """Smooth transitions using easing functions"""

    @staticmethod
    def ease_in_out_cubic(t):
        """Smooth acceleration and deceleration"""
        if t < 0.5:
            return 4 * t**3
        else:
            return 1 - ((-2 * t + 2)**3) / 2

    @staticmethod
    def ease_out_expo(t):
        """Fast start, slow stop"""
        if t >= 1:
            return 1
        return 1 - 2**(-10 * t)

    @staticmethod
    def ease_in_out_back(t):
        """Slight overshoot for snappy feel"""
        c1 = 1.70158
        c2 = c1 * 1.525

        if t < 0.5:
            return ((2 * t)**2 * ((c2 + 1) * 2 * t - c2)) / 2
        else:
            return ((2 * t - 2)**2 * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2

class SmoothMotion:
    def __init__(self, transition_time=0.2):
        self.transition_time = transition_time
        self.current_position = np.array([0.0, 0.0])
        self.target_position = np.array([0.0, 0.0])
        self.start_position = np.array([0.0, 0.0])
        self.elapsed_time = 0
        self.in_transition = False

    def set_target(self, target):
        """Begin smooth transition to target"""
        self.start_position = self.current_position.copy()
        self.target_position = np.array(target)
        self.elapsed_time = 0
        self.in_transition = True

    def update(self, dt):
        """Update position with easing"""
        if not self.in_transition:
            return self.current_position

        self.elapsed_time += dt
        progress = min(1.0, self.elapsed_time / self.transition_time)

        # Apply easing curve
        eased_progress = EaseCurve.ease_in_out_cubic(progress)

        # Interpolate position
        self.current_position = self.start_position + \
                               (self.target_position - self.start_position) * \
                               eased_progress

        if progress >= 1.0:
            self.in_transition = False

        return self.current_position
```

**Velocity-Based Smoothing**:
```python
class VelocitySmoothing:
    """Smooth cursor using velocity constraints"""

    def __init__(self, max_velocity=1000, acceleration=3000,
                 deceleration=5000):
        self.max_velocity = max_velocity        # pixels/second
        self.acceleration = acceleration        # pixels/second²
        self.deceleration = deceleration        # pixels/second²
        self.velocity = np.array([0.0, 0.0])
        self.position = np.array([0.0, 0.0])

    def update(self, target_position, dt):
        """Move towards target with velocity constraints"""
        direction = target_position - self.position
        distance = np.linalg.norm(direction)

        if distance < 1.0:
            # Close enough, snap to target
            self.position = target_position
            self.velocity *= 0
            return self.position

        direction_normalized = direction / distance

        # Calculate desired velocity
        desired_speed = min(self.max_velocity, distance / dt)
        desired_velocity = direction_normalized * desired_speed

        # Determine acceleration or deceleration
        velocity_delta = desired_velocity - self.velocity
        delta_magnitude = np.linalg.norm(velocity_delta)

        if delta_magnitude > 0:
            delta_direction = velocity_delta / delta_magnitude

            # Apply acceleration limit
            max_delta = self.acceleration * dt
            if delta_magnitude > max_delta:
                velocity_delta = delta_direction * max_delta

            self.velocity += velocity_delta

        # Update position
        self.position += self.velocity * dt

        return self.position
```

**Recommended Settings**:
- Max Acceleration: 3000-5000 pixels/s²
- Max Deceleration: 5000-8000 pixels/s² (faster stops feel better)
- Transition Time: 0.15-0.25 seconds
- Easing: ease_in_out_cubic for natural feel

---

### 4.4 Jitter Reduction (Smoothing Algorithms)

**Purpose**: Remove high-frequency noise and hand tremor without adding latency.

**1. Exponential Moving Average** (best for real-time):
```python
class ExponentialSmoothing:
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # 0 = max smoothing, 1 = no smoothing
        self.smoothed_x = None
        self.smoothed_y = None

    def smooth(self, x, y):
        """Apply EMA smoothing"""
        if self.smoothed_x is None:
            # Initialize
            self.smoothed_x = x
            self.smoothed_y = y
        else:
            # Update
            self.smoothed_x = self.alpha * x + (1 - self.alpha) * self.smoothed_x
            self.smoothed_y = self.alpha * y + (1 - self.alpha) * self.smoothed_y

        return self.smoothed_x, self.smoothed_y
```

**2. Double Exponential Smoothing** (for trending data):
```python
class DoubleExponentialSmoothing:
    def __init__(self, alpha=0.3, beta=0.1):
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Trend smoothing
        self.level = None
        self.trend = None

    def smooth(self, value):
        """Apply double exponential smoothing"""
        if self.level is None:
            self.level = value
            self.trend = 0
            return value

        # Update level
        last_level = self.level
        self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)

        # Update trend
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend

        # Forecast next value
        forecast = self.level + self.trend

        return forecast
```

**3. One Euro Filter** (velocity-adaptive smoothing):
```python
class OneEuroFilter:
    """
    Adaptive filter that reduces jitter while maintaining responsiveness
    Low-pass filter with cutoff frequency that adapts to velocity
    """

    def __init__(self, freq=100, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq = freq          # Sampling frequency
        self.mincutoff = mincutoff  # Minimum cutoff frequency
        self.beta = beta           # Cutoff slope
        self.dcutoff = dcutoff     # Cutoff for derivative

        self.x_prev = None
        self.dx_prev = 0

    def smoothing_factor(self, cutoff):
        """Calculate smoothing factor from cutoff frequency"""
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        """Apply One Euro filter"""
        if self.x_prev is None:
            self.x_prev = x
            return x

        # Calculate derivative
        dx = (x - self.x_prev) * self.freq

        # Smooth derivative
        alpha_d = self.smoothing_factor(self.dcutoff)
        dx_smoothed = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Calculate adaptive cutoff
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)

        # Smooth value
        alpha = self.smoothing_factor(cutoff)
        x_filtered = alpha * x + (1 - alpha) * self.x_prev

        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_smoothed

        return x_filtered

class TwoDimensionalOneEuroFilter:
    """One Euro Filter for 2D cursor position"""

    def __init__(self, freq=100, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.filter_x = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.filter_y = OneEuroFilter(freq, mincutoff, beta, dcutoff)

    def filter(self, x, y):
        """Filter 2D position"""
        x_filtered = self.filter_x.filter(x)
        y_filtered = self.filter_y.filter(y)
        return x_filtered, y_filtered
```

**4. Savitzky-Golay Filter** (for smooth derivative):
```python
from scipy.signal import savgol_filter

class SavitzkyGolaySmoothing:
    def __init__(self, window_length=11, polyorder=3):
        self.window_length = window_length  # Must be odd
        self.polyorder = polyorder
        self.history = []

    def smooth(self, value):
        """Apply Savitzky-Golay smoothing"""
        self.history.append(value)

        if len(self.history) < self.window_length:
            # Not enough history yet
            return value

        # Keep only needed history
        if len(self.history) > self.window_length:
            self.history.pop(0)

        # Apply filter
        smoothed = savgol_filter(self.history, self.window_length,
                                self.polyorder)

        # Return center value (current)
        return smoothed[-1]
```

**Comparison & Recommendations**:

| Filter | Latency | Jitter Reduction | Responsiveness | Best For |
|--------|---------|------------------|----------------|----------|
| **EMA** | Very Low | Good | Excellent | Real-time cursor, gaming |
| **Double EMA** | Low | Better | Good | Trending motion |
| **One Euro** | Low | Excellent | Excellent | **Recommended for cursor** |
| **Savitzky-Golay** | Medium | Excellent | Good | Post-processing, smoothness priority |

**Recommended**: **One Euro Filter** for cursor control
- Adapts to velocity (fast movements = less filtering)
- Low latency
- Excellent jitter reduction
- Used in VR/AR tracking systems

**Parameter Tuning**:
- **mincutoff** (1.0-3.0): Lower = smoother, higher = more responsive
  - Cursor: 1.0-1.5
  - Gaming: 2.0-3.0
- **beta** (0.001-0.01): Controls adaptation speed
  - Start with 0.007
  - Increase if cursor feels sluggish
- **freq**: Match your IMU sampling rate (50-100Hz)

---

### 4.5 Edge Snapping and Target Assistance

**Purpose**: Help users hit targets and screen edges more easily.

**1. Edge Snapping**:
```python
class EdgeSnapping:
    def __init__(self, snap_zone=50, snap_strength=0.8):
        self.snap_zone = snap_zone      # pixels from edge
        self.snap_strength = snap_strength  # 0-1, how much to snap

    def apply(self, x, y, screen_width, screen_height):
        """Snap cursor to edges within snap zone"""
        # Left edge
        if x < self.snap_zone:
            blend = 1 - (x / self.snap_zone)  # 1 at edge, 0 at zone boundary
            x = x * (1 - blend * self.snap_strength)

        # Right edge
        elif x > screen_width - self.snap_zone:
            blend = (x - (screen_width - self.snap_zone)) / self.snap_zone
            target_x = screen_width
            x = x + (target_x - x) * blend * self.snap_strength

        # Top edge
        if y < self.snap_zone:
            blend = 1 - (y / self.snap_zone)
            y = y * (1 - blend * self.snap_strength)

        # Bottom edge
        elif y > screen_height - self.snap_zone:
            blend = (y - (screen_height - self.snap_zone)) / self.snap_zone
            target_y = screen_height
            y = y + (target_y - y) * blend * self.snap_strength

        return x, y
```

**2. Target Assistance (Aim Assist)**:
```python
class TargetAssistance:
    """Magnetic cursor that's attracted to UI elements"""

    def __init__(self, attraction_radius=100, attraction_strength=0.3):
        self.attraction_radius = attraction_radius
        self.attraction_strength = attraction_strength
        self.targets = []  # List of (x, y, radius) tuples

    def set_targets(self, ui_elements):
        """Set current clickable targets (buttons, links, etc.)"""
        self.targets = []
        for element in ui_elements:
            # Extract center and size of clickable elements
            cx = element['x'] + element['width'] / 2
            cy = element['y'] + element['height'] / 2
            radius = max(element['width'], element['height']) / 2
            self.targets.append((cx, cy, radius))

    def apply(self, cursor_x, cursor_y, velocity_magnitude):
        """Apply magnetic attraction to nearby targets"""
        if velocity_magnitude > 500:  # Don't assist during fast movements
            return cursor_x, cursor_y

        # Find nearest target
        nearest_target = None
        nearest_distance = float('inf')

        for tx, ty, tr in self.targets:
            distance = math.sqrt((cursor_x - tx)**2 + (cursor_y - ty)**2)

            # Check if within attraction radius
            if distance < self.attraction_radius + tr:
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_target = (tx, ty, tr)

        if nearest_target:
            tx, ty, tr = nearest_target

            # Calculate attraction strength (stronger when closer)
            normalized_distance = nearest_distance / self.attraction_radius
            strength = self.attraction_strength * (1 - normalized_distance)

            # Reduce strength based on velocity (less assist when moving fast)
            velocity_factor = max(0, 1 - velocity_magnitude / 500)
            strength *= velocity_factor

            # Apply attraction
            dx = tx - cursor_x
            dy = ty - cursor_y
            cursor_x += dx * strength
            cursor_y += dy * strength

        return cursor_x, cursor_y
```

**3. Corner Snapping** (for window management):
```python
class CornerSnapping:
    """Snap to screen corners for window tiling gestures"""

    def __init__(self, corner_zone=150):
        self.corner_zone = corner_zone

    def detect_corner(self, x, y, screen_width, screen_height):
        """Detect if cursor is in a corner zone"""
        in_left = x < self.corner_zone
        in_right = x > screen_width - self.corner_zone
        in_top = y < self.corner_zone
        in_bottom = y > screen_height - self.corner_zone

        if in_top and in_left:
            return 'top-left', (0, 0)
        elif in_top and in_right:
            return 'top-right', (screen_width, 0)
        elif in_bottom and in_left:
            return 'bottom-left', (0, screen_height)
        elif in_bottom and in_right:
            return 'bottom-right', (screen_width, screen_height)

        return None, None
```

**Usage Recommendations**:
- **Edge Snapping**: Enable for general use, 50px zone, 0.6-0.8 strength
- **Target Assistance**: Optional, useful for accessibility, 0.2-0.4 strength
- **Corner Snapping**: Enable for window management, 150px zone

---

### 4.6 Complete UX Pipeline

**Integrated System**:
```python
class CursorUXPipeline:
    """Complete UX processing pipeline"""

    def __init__(self):
        self.dead_zone = RadialDeadZone(radius=2.0)
        self.sensitivity = PowerCurveSensitivity(
            base_sensitivity=2.0,
            power=2.0,
            knee=0.5
        )
        self.smoothing = TwoDimensionalOneEuroFilter(
            freq=100,
            mincutoff=1.2,
            beta=0.007
        )
        self.acceleration = VelocitySmoothing(
            max_velocity=1200,
            acceleration=4000,
            deceleration=6000
        )
        self.edge_snap = EdgeSnapping(snap_zone=50, snap_strength=0.7)
        self.target_assist = TargetAssistance(
            attraction_radius=80,
            attraction_strength=0.25
        )

    def process(self, raw_roll, raw_pitch, dt, ui_targets=None):
        """Complete processing pipeline"""
        # 1. Apply dead zone
        roll, pitch = self.dead_zone.apply(raw_roll, raw_pitch)

        # 2. Apply sensitivity curve
        velocity_x = self.sensitivity.apply(roll)
        velocity_y = self.sensitivity.apply(pitch)

        # 3. Smooth velocities
        velocity_x, velocity_y = self.smoothing.filter(velocity_x, velocity_y)

        # 4. Apply acceleration limits
        velocity = np.array([velocity_x, velocity_y])
        target_position = current_position + velocity * dt
        smoothed_position = self.acceleration.update(target_position, dt)

        # 5. Apply edge snapping
        x, y = self.edge_snap.apply(
            smoothed_position[0],
            smoothed_position[1],
            screen_width,
            screen_height
        )

        # 6. Apply target assistance (if UI targets provided)
        if ui_targets:
            self.target_assist.set_targets(ui_targets)
            velocity_mag = np.linalg.norm(velocity)
            x, y = self.target_assist.apply(x, y, velocity_mag)

        return int(x), int(y)
```

**Tuning for Different Scenarios**:

**Gaming** (fast, responsive):
```python
gaming_pipeline = CursorUXPipeline()
gaming_pipeline.dead_zone.radius = 1.5         # Smaller
gaming_pipeline.sensitivity.exponent = 2.5     # More aggressive
gaming_pipeline.smoothing.mincutoff = 2.5      # Less smoothing
gaming_pipeline.target_assist = None           # No assist
```

**Productivity** (balanced):
```python
productivity_pipeline = CursorUXPipeline()
productivity_pipeline.dead_zone.radius = 2.0   # Standard
productivity_pipeline.sensitivity.power = 2.0  # Balanced
productivity_pipeline.smoothing.mincutoff = 1.2  # Good smoothing
productivity_pipeline.target_assist.strength = 0.25  # Subtle assist
```

**Accessibility** (stable, assisted):
```python
accessibility_pipeline = CursorUXPipeline()
accessibility_pipeline.dead_zone.radius = 3.0    # Larger
accessibility_pipeline.sensitivity.power = 1.5   # Gentler
accessibility_pipeline.smoothing.mincutoff = 0.8 # Heavy smoothing
accessibility_pipeline.target_assist.strength = 0.4  # Strong assist
accessibility_pipeline.target_assist.radius = 120  # Larger radius
```

---

## 5. Gesture Detection

Detecting finger pinches, taps, and other gestures from IMU data for click events and commands.

### 5.1 Understanding Gesture Signatures

**Finger-Pinch Detection Principle**:
- When fingers touch, creates vibration spike in accelerometer
- Characteristic frequency: 100-400 Hz
- Magnitude: 0.5-3.0 g (varies by force)
- Duration: 10-50 ms

**Key Features**:
1. **Accelerometer magnitude spike**
2. **Short duration pulse**
3. **Distinct from hand motion** (higher frequency)
4. **Orientation context** (gyroscope helps distinguish type)

---

### 5.2 Basic Tap Detection

**Simple Threshold-Based**:
```python
class SimpleTapDetector:
    def __init__(self, threshold=1.5, cooldown=0.3):
        self.threshold = threshold  # g (gravitational acceleration)
        self.cooldown = cooldown    # seconds between taps
        self.last_tap_time = 0

    def detect(self, accel_x, accel_y, accel_z, timestamp):
        """Detect tap from accelerometer spike"""
        # Calculate magnitude
        magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Check cooldown
        if timestamp - self.last_tap_time < self.cooldown:
            return False

        # Detect spike above threshold
        if magnitude > self.threshold + 1.0:  # +1.0 for gravity baseline
            self.last_tap_time = timestamp
            return True

        return False
```

**Enhanced Tap Detection with Frequency Analysis**:
```python
class EnhancedTapDetector:
    def __init__(self, sample_rate=100, threshold=1.2,
                 min_freq=50, max_freq=400):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.accel_history = []
        self.window_size = int(sample_rate * 0.1)  # 100ms window
        self.cooldown = 0.2  # seconds
        self.last_tap_time = 0

    def detect(self, accel_x, accel_y, accel_z, timestamp):
        """Detect tap using magnitude and frequency analysis"""
        # Calculate magnitude
        magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Add to history
        self.accel_history.append(magnitude)
        if len(self.accel_history) > self.window_size:
            self.accel_history.pop(0)

        # Check cooldown
        if timestamp - self.last_tap_time < self.cooldown:
            return False

        # Need enough history
        if len(self.accel_history) < self.window_size:
            return False

        # Calculate baseline (should be ~1g with some motion)
        baseline = np.median(self.accel_history)

        # Current spike above baseline?
        spike = magnitude - baseline

        if spike > self.threshold:
            # Check if spike is short duration (tap signature)
            # Count how many recent samples are elevated
            recent_elevated = sum(1 for m in self.accel_history[-20:]
                                 if m - baseline > self.threshold * 0.5)

            # Tap should be brief (5-15 samples at 100Hz = 50-150ms)
            if 3 < recent_elevated < 15:
                self.last_tap_time = timestamp
                return True

        return False
```

---

### 5.3 Double-Tap Detection

```python
class DoubleTapDetector:
    def __init__(self, tap_detector, max_interval=0.5):
        self.tap_detector = tap_detector
        self.max_interval = max_interval  # Max time between taps
        self.first_tap_time = None

    def detect(self, accel_x, accel_y, accel_z, timestamp):
        """Detect double-tap gesture"""
        # Detect single tap
        if self.tap_detector.detect(accel_x, accel_y, accel_z, timestamp):
            if self.first_tap_time is None:
                # First tap
                self.first_tap_time = timestamp
                return 'single_tap'
            else:
                # Check if second tap is within interval
                interval = timestamp - self.first_tap_time
                if interval < self.max_interval:
                    # Double tap detected!
                    self.first_tap_time = None
                    return 'double_tap'
                else:
                    # Too slow, restart
                    self.first_tap_time = timestamp
                    return 'single_tap'

        # Timeout first tap if too much time passes
        if self.first_tap_time and timestamp - self.first_tap_time > self.max_interval:
            self.first_tap_time = None

        return None
```

---

### 5.4 Hold Detection (Long Press)

```python
class HoldDetector:
    def __init__(self, tap_detector, hold_duration=0.5):
        self.tap_detector = tap_detector
        self.hold_duration = hold_duration
        self.hold_start_time = None
        self.hold_detected = False

    def detect(self, accel_x, accel_y, accel_z, timestamp):
        """Detect hold gesture (sustained pinch)"""
        magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Check if currently in elevated state (fingers together)
        baseline = 1.0  # Approximate gravity
        is_elevated = magnitude > baseline + 0.5

        if is_elevated:
            if self.hold_start_time is None:
                # Start of potential hold
                self.hold_start_time = timestamp
            else:
                # Check if held long enough
                hold_time = timestamp - self.hold_start_time
                if hold_time > self.hold_duration and not self.hold_detected:
                    self.hold_detected = True
                    return 'hold_start'
        else:
            # Released
            if self.hold_detected:
                self.hold_detected = False
                self.hold_start_time = None
                return 'hold_end'
            self.hold_start_time = None

        return None
```

---

### 5.5 Gesture State Machine

**Finite State Machine for Robust Recognition**:
```python
class GestureStateMachine:
    """
    State machine for recognizing gesture sequences
    States: IDLE -> CONTACT -> TAP/HOLD/DRAG
    """

    def __init__(self):
        self.state = 'IDLE'
        self.contact_start_time = None
        self.first_tap_time = None
        self.tap_threshold = 1.5  # g
        self.hold_threshold = 0.5  # seconds
        self.double_tap_window = 0.5  # seconds

    def update(self, accel_magnitude, timestamp):
        """Update state machine"""
        baseline = 1.0
        is_contact = accel_magnitude > baseline + self.tap_threshold

        event = None

        if self.state == 'IDLE':
            if is_contact:
                self.state = 'CONTACT'
                self.contact_start_time = timestamp

        elif self.state == 'CONTACT':
            contact_duration = timestamp - self.contact_start_time

            if not is_contact:
                # Contact released
                if contact_duration < 0.15:  # Quick tap
                    # Check for double tap
                    if self.first_tap_time:
                        interval = timestamp - self.first_tap_time
                        if interval < self.double_tap_window:
                            event = 'double_tap'
                            self.first_tap_time = None
                        else:
                            event = 'tap'
                            self.first_tap_time = timestamp
                    else:
                        event = 'tap'
                        self.first_tap_time = timestamp
                else:
                    # Held then released
                    event = 'hold_end'

                self.state = 'IDLE'
                self.contact_start_time = None

            elif contact_duration > self.hold_threshold:
                # Transition to hold
                self.state = 'HOLD'
                event = 'hold_start'

        elif self.state == 'HOLD':
            if not is_contact:
                self.state = 'IDLE'
                self.contact_start_time = None
                event = 'hold_end'

        # Clear first tap if too old
        if self.first_tap_time and timestamp - self.first_tap_time > self.double_tap_window:
            self.first_tap_time = None

        return event, self.state
```

**Usage**:
```python
fsm = GestureStateMachine()

while True:
    accel, gyro = read_imu()
    magnitude = np.linalg.norm(accel)

    event, state = fsm.update(magnitude, time.time())

    if event == 'tap':
        print("Single tap - LEFT CLICK")
        perform_left_click()
    elif event == 'double_tap':
        print("Double tap - DOUBLE CLICK")
        perform_double_click()
    elif event == 'hold_start':
        print("Hold start - BEGIN DRAG")
        mouse_down()
    elif event == 'hold_end':
        print("Hold end - END DRAG")
        mouse_up()
```

---

### 5.6 Advanced Gesture Recognition

**Palm Orientation Detection**:
```python
class PalmOrientationDetector:
    """Detect palm up/down/forward for mode switching"""

    def __init__(self):
        self.pitch_threshold = 45  # degrees

    def detect_orientation(self, pitch):
        """Determine palm orientation"""
        if pitch < -self.pitch_threshold:
            return 'palm_up'
        elif pitch > self.pitch_threshold:
            return 'palm_down'
        else:
            return 'palm_forward'
```

**Wrist Flick Detection**:
```python
class WristFlickDetector:
    """Detect quick wrist rotation for scrolling"""

    def __init__(self, velocity_threshold=200, duration_threshold=0.2):
        self.velocity_threshold = velocity_threshold  # deg/s
        self.duration_threshold = duration_threshold  # seconds
        self.flick_start_time = None
        self.flick_direction = None

    def detect(self, gyro_z, timestamp):
        """Detect wrist flick (rotation around Z axis)"""
        is_flicking = abs(gyro_z) > self.velocity_threshold

        if is_flicking:
            if self.flick_start_time is None:
                # Start of flick
                self.flick_start_time = timestamp
                self.flick_direction = 'right' if gyro_z > 0 else 'left'
        else:
            if self.flick_start_time is not None:
                # End of flick
                duration = timestamp - self.flick_start_time

                if duration < self.duration_threshold:
                    # Valid flick detected
                    direction = self.flick_direction
                    self.flick_start_time = None
                    self.flick_direction = None
                    return f'flick_{direction}'

                self.flick_start_time = None
                self.flick_direction = None

        return None
```

**Shake Detection**:
```python
class ShakeDetector:
    """Detect shake gesture for undo/reset"""

    def __init__(self, threshold=2.5, min_shakes=3, time_window=1.0):
        self.threshold = threshold  # g
        self.min_shakes = min_shakes
        self.time_window = time_window
        self.shake_times = []

    def detect(self, accel_x, accel_y, accel_z, timestamp):
        """Detect shake gesture"""
        # Calculate total acceleration (subtract gravity)
        magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        lateral = abs(magnitude - 1.0)  # Deviation from 1g

        if lateral > self.threshold:
            self.shake_times.append(timestamp)

        # Remove old shakes outside time window
        self.shake_times = [t for t in self.shake_times
                           if timestamp - t < self.time_window]

        # Check if enough shakes in window
        if len(self.shake_times) >= self.min_shakes:
            self.shake_times = []  # Reset
            return True

        return False
```

---

### 5.7 Context-Aware Gesture Recognition

**Combining Gestures with Orientation**:
```python
class ContextualGestureRecognizer:
    """Recognize gestures that depend on hand orientation"""

    def __init__(self):
        self.tap_detector = EnhancedTapDetector()
        self.palm_detector = PalmOrientationDetector()
        self.flick_detector = WristFlickDetector()

    def recognize(self, accel, gyro, orientation, timestamp):
        """Recognize gesture with orientation context"""
        accel_x, accel_y, accel_z = accel
        gyro_x, gyro_y, gyro_z = gyro
        roll, pitch, yaw = orientation

        # Determine palm orientation
        palm_orientation = self.palm_detector.detect_orientation(pitch)

        # Check for tap
        if self.tap_detector.detect(accel_x, accel_y, accel_z, timestamp):
            # Contextualize tap based on palm orientation
            if palm_orientation == 'palm_up':
                return 'back_gesture'  # Like WowMouse
            else:
                return 'tap'

        # Check for flick
        flick = self.flick_detector.detect(gyro_z, timestamp)
        if flick:
            if palm_orientation == 'palm_down':
                # Palm down flick -> scroll
                return f'scroll_{flick.split("_")[1]}'
            else:
                # Palm forward flick -> navigate
                return f'navigate_{flick.split("_")[1]}'

        return None
```

---

### 5.8 Machine Learning Approach (Optional)

**For more robust recognition with personalization**:

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class MLGestureRecognizer:
    """
    Machine learning-based gesture recognition
    Learns from examples, more robust to individual differences
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.window_size = 50  # samples (0.5s at 100Hz)
        self.feature_buffer = []
        self.is_trained = False

    def extract_features(self, accel_window, gyro_window):
        """Extract features from sensor windows"""
        accel_array = np.array(accel_window)
        gyro_array = np.array(gyro_window)

        features = []

        # Statistical features
        for arr in [accel_array, gyro_array]:
            features.extend([
                np.mean(arr, axis=0),
                np.std(arr, axis=0),
                np.min(arr, axis=0),
                np.max(arr, axis=0),
                np.ptp(arr, axis=0),  # Peak-to-peak
            ])

        # Frequency domain features (simplified)
        accel_magnitude = np.linalg.norm(accel_array, axis=1)
        fft = np.fft.fft(accel_magnitude)
        features.append(np.abs(fft[:10]))  # First 10 frequency bins

        return np.concatenate([f.flatten() if hasattr(f, 'flatten') else [f]
                              for f in features])

    def train(self, training_data, labels):
        """
        Train model on labeled gesture examples
        training_data: list of (accel_window, gyro_window) tuples
        labels: list of gesture labels ('tap', 'double_tap', 'hold', etc.)
        """
        X = []
        for accel_window, gyro_window in training_data:
            features = self.extract_features(accel_window, gyro_window)
            X.append(features)

        self.model.fit(X, labels)
        self.is_trained = True
        print(f"Model trained on {len(X)} examples")

    def predict(self, accel_window, gyro_window):
        """Predict gesture from sensor window"""
        if not self.is_trained:
            return None

        features = self.extract_features(accel_window, gyro_window)
        prediction = self.model.predict([features])[0]
        confidence = np.max(self.model.predict_proba([features]))

        # Only return prediction if confident
        if confidence > 0.7:
            return prediction
        return None
```

**Collection Training Data**:
```python
def collect_training_data():
    """Interactive training data collection"""
    training_data = []
    labels = []

    gestures = ['tap', 'double_tap', 'hold', 'flick_left', 'flick_right']

    for gesture in gestures:
        print(f"\nPerform {gesture} gesture 10 times...")
        for i in range(10):
            print(f"  Repetition {i+1}/10 - Ready...")
            time.sleep(1)
            print("  GO!")

            # Collect 0.5 seconds of data
            accel_window = []
            gyro_window = []
            start_time = time.time()

            while time.time() - start_time < 0.5:
                accel, gyro = read_imu()
                accel_window.append(accel)
                gyro_window.append(gyro)
                time.sleep(0.01)  # 100Hz

            training_data.append((accel_window, gyro_window))
            labels.append(gesture)
            print("  Captured!")

    return training_data, labels

# Train model
ml_recognizer = MLGestureRecognizer()
training_data, labels = collect_training_data()
ml_recognizer.train(training_data, labels)
```

---

### 5.9 Gesture Recognition Best Practices

**Implementation Recommendations**:

1. **Start Simple**: Begin with threshold-based tap detection
2. **Add State Machine**: Implement FSM for tap/double-tap/hold
3. **Tune Parameters**: Adjust thresholds per device and user
4. **Provide Feedback**: Haptic/visual confirmation of gestures
5. **Consider ML**: For production systems with diverse users

**Tuning Guidelines**:
- **Tap Threshold**: 1.2-1.8g (lower = more sensitive)
- **Hold Duration**: 0.4-0.6s (shorter = faster response)
- **Double-Tap Window**: 0.4-0.6s
- **Cooldown**: 0.2-0.3s between gestures

**Robustness Strategies**:
- Use state machines to prevent spurious detections
- Combine multiple sensor signals (accel + gyro)
- Add orientation context
- Implement cooldown periods
- Provide user feedback for confirmation

---

## 6. Integration Recommendations

### 6.1 Complete System Architecture

**Recommended Architecture**:
```
IMU Sensors (100Hz)
    ↓
Sensor Fusion (Madgwick Filter)
    ↓
Orientation Quaternion
    ↓
├─→ Gesture Recognition (FSM)
│       ↓
│   Gesture Events (tap, double-tap, hold)
│       ↓
│   Mouse Click Events
│
└─→ Positioning Algorithm (Relative/Absolute/Hybrid)
        ↓
    Dead Zone Filter
        ↓
    Sensitivity Curve
        ↓
    One Euro Smoothing
        ↓
    Acceleration Limiting
        ↓
    Edge Snapping
        ↓
    Target Assistance
        ↓
    Cursor Position
        ↓
    OS Cursor Control
```

---

### 6.2 Implementation Roadmap

**Phase 1: Foundation (Week 1-2)**
1. Implement Madgwick filter for orientation
2. Basic complementary filter as fallback
3. Simple relative positioning
4. Threshold-based tap detection

**Phase 2: Core Features (Week 3-4)**
5. Enhanced UX pipeline (dead zone, sensitivity, smoothing)
6. Gesture state machine (tap, double-tap, hold)
7. Drift correction (ZUPT, periodic recalibration)
8. Basic absolute positioning with calibration

**Phase 3: Refinement (Week 5-6)**
9. Hybrid positioning modes
10. Advanced gesture detection (flick, shake, palm orientation)
11. Adaptive parameters
12. Target assistance

**Phase 4: Polish (Week 7-8)**
13. Per-application profiles
14. ML gesture recognition (optional)
15. Performance optimization
16. User testing and tuning

---

### 6.3 Performance Targets

**Latency Budget**:
- IMU sampling: 10ms (100Hz)
- Sensor fusion: <5ms
- Gesture detection: <5ms
- Cursor algorithm: <5ms
- UX pipeline: <5ms
- OS injection: <10ms
- **Total: <40ms** (target), <60ms (acceptable)

**Accuracy Targets**:
- Orientation accuracy: <2° RMS error
- Cursor jitter: <5 pixels (stationary)
- Gesture recognition: >95% accuracy
- False positive rate: <5% (1 per 20 samples)

---

### 6.4 Configuration System

**User-Adjustable Parameters**:
```python
class CursorControlConfig:
    """Configuration management for cursor control"""

    def __init__(self):
        # Sensor fusion
        self.fusion_algorithm = 'madgwick'  # 'madgwick', 'mahony', 'complementary'
        self.fusion_beta = 0.1

        # Positioning
        self.positioning_mode = 'relative'  # 'relative', 'absolute', 'hybrid'
        self.sensitivity = 2.0
        self.max_speed = 1200

        # Dead zone
        self.dead_zone_radius = 2.0

        # Smoothing
        self.smoothing_alpha = 0.3  # EMA
        self.smoothing_mincutoff = 1.2  # One Euro

        # Gestures
        self.tap_threshold = 1.5
        self.hold_duration = 0.5
        self.double_tap_window = 0.5

        # Features
        self.enable_edge_snap = True
        self.enable_target_assist = False
        self.target_assist_strength = 0.25

    def load_profile(self, profile_name):
        """Load predefined profile"""
        profiles = {
            'gaming': {
                'sensitivity': 3.0,
                'dead_zone_radius': 1.5,
                'smoothing_alpha': 0.5,
                'enable_target_assist': False,
            },
            'productivity': {
                'sensitivity': 2.0,
                'dead_zone_radius': 2.0,
                'smoothing_alpha': 0.3,
                'enable_target_assist': True,
            },
            'accessibility': {
                'sensitivity': 1.5,
                'dead_zone_radius': 3.0,
                'smoothing_alpha': 0.2,
                'enable_target_assist': True,
                'target_assist_strength': 0.4,
            }
        }

        if profile_name in profiles:
            for key, value in profiles[profile_name].items():
                setattr(self, key, value)

    def save(self, filename):
        """Save configuration to file"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def load(self, filename):
        """Load configuration from file"""
        import json
        with open(filename, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(self, key, value)
```

---

### 6.5 Testing and Validation

**Unit Tests**:
```python
def test_madgwick_filter():
    """Test sensor fusion accuracy"""
    filter = MadgwickFilter()

    # Simulate stationary device (only gravity)
    for i in range(100):
        accel = [0, 0, 1.0]  # 1g downward
        gyro = [0, 0, 0]     # No rotation
        filter.update_imu(accel, gyro)

    roll, pitch, yaw = filter.get_euler_angles()

    # Should be level (roll ≈ 0, pitch ≈ 0)
    assert abs(roll) < 0.1  # Within 0.1 radians
    assert abs(pitch) < 0.1

def test_tap_detection():
    """Test gesture detection"""
    detector = EnhancedTapDetector()

    # Simulate tap (spike in accelerometer)
    timestamps = np.linspace(0, 1, 100)
    for t in timestamps:
        if 0.5 < t < 0.55:  # 50ms spike
            accel = [0, 0, 3.0]  # Strong spike
        else:
            accel = [0, 0, 1.0]  # Normal gravity

        tap = detector.detect(accel[0], accel[1], accel[2], t)
        if tap:
            print(f"Tap detected at {t}s")
            assert 0.5 < t < 0.6  # Should detect during spike

def test_cursor_movement():
    """Test cursor positioning"""
    cursor = EnhancedRelativePositioning()

    # Simulate tilting right (positive roll)
    for i in range(50):
        x, y = cursor.update(roll=10, pitch=0, dt=0.01)

    # Cursor should have moved right
    assert x > 0

# Run tests
test_madgwick_filter()
test_tap_detection()
test_cursor_movement()
print("All tests passed!")
```

**Performance Benchmarking**:
```python
import time

def benchmark_sensor_fusion(filter, iterations=1000):
    """Benchmark sensor fusion performance"""
    accel = [0, 0, 1.0]
    gyro = [0.1, 0.2, 0.05]

    start = time.perf_counter()
    for i in range(iterations):
        filter.update_imu(accel, gyro)
    end = time.perf_counter()

    total_time = (end - start) * 1000  # Convert to ms
    avg_time = total_time / iterations

    print(f"Average time per update: {avg_time:.3f}ms")
    print(f"Max frequency: {1000/avg_time:.1f}Hz")

# Benchmark different filters
print("Madgwick Filter:")
benchmark_sensor_fusion(MadgwickFilter())

print("\nMahony Filter:")
benchmark_sensor_fusion(MahonyFilter())

print("\nComplementary Filter:")
benchmark_sensor_fusion(ComplementaryFilter())
```

---

## 7. References

### 7.1 Research Papers & Documentation

**Sensor Fusion:**
- [x-io Technologies - Open Source IMU and AHRS Algorithms](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)
- [AHRS Python Library Documentation](https://ahrs.readthedocs.io/en/latest/filters/madgwick.html)
- [Madgwick Internal Report - Efficient Orientation Filter](https://courses.cs.washington.edu/courses/cse466/14au/labs/l4/madgwick_internal_report.pdf)
- [OlliW's IMU Data Fusing: Complementary, Kalman, and Mahony Filter](https://www.olliw.eu/2013/imu-data-fusing/)

**Cursor Control:**
- [PMC - Cursor Control by Kalman Filter with Non-Invasive Body-Machine Interface](https://pmc.ncbi.nlm.nih.gov/articles/PMC4341977/)
- [Sparx Engineering - IMU Signal Processing with Kalman Filter](https://sparxeng.com/blog/software/imu-signal-processing-with-kalman-filter)
- [arXiv - Using Inertial Sensors for Position and Orientation Estimation](https://arxiv.org/pdf/1704.06053)

**Gesture Recognition:**
- [ResearchGate - Serendipity: Finger Gesture Recognition using Off-the-Shelf Smartwatch](https://www.researchgate.net/publication/301931139_Serendipity_Finger_Gesture_Recognition_using_an_Off-the-Shelf_Smartwatch)
- [Carnegie Mellon - ViBand: Repurposed Sensor for Bio-Acoustic Signals](https://www.cmu.edu/news/stories/archives/2016/october/smartwatch-capability.html)
- [IEEE - Gesture Modeling and Recognition Using Finite State Machines](https://ieeexplore.ieee.org/document/840667/)

**Drift Correction:**
- [Semantic Scholar - Gyroscope Drift Correction Algorithm for IMU](https://www.semanticscholar.org/paper/Gyroscope-Drift-Correction-Algorithm-for-Inertial-Tangnimitchok-Barreto/bc9024e573c6fa30b622286c7d7f460e4eb6a693)
- [NCBI - Magnetometer-Based Drift Correction During Rest in IMU](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6471153/)

**UX & Filtering:**
- [Makeability Lab - Smoothing Input for Physical Computing](https://makeabilitylab.github.io/physcomp/advancedio/smoothing-input.html)
- [mbedded.ninja - Exponential Moving Average (EMA) Filters](https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/)
- [UMA Technology - Controller Deadzone Settings](https://umatechnology.org/controller-deadzone-settings/)

### 7.2 Implementation Resources

**Open Source Libraries:**
- [GitHub - bjohnsonfl/Madgwick_Filter](https://github.com/bjohnsonfl/Madgwick_Filter) - C/C++ implementation
- [GitHub - vishwas1101/Filters](https://github.com/vishwas1101/Filters) - Arduino complementary filter implementations
- [AHRS Python Library](https://ahrs.readthedocs.io/) - Comprehensive orientation algorithms
- [GitHub - jonnieZG/EWMA](https://github.com/jonnieZG/EWMA) - Exponential smoothing

**Development Tools:**
- [MATLAB Sensor Fusion Toolbox](https://www.mathworks.com/help/fusion/)
- [SimpleFusion Arduino Library](https://seanboe.com/blog/complementary-filters)

### 7.3 Commercial References

**DoublePoint WowMouse:**
- [Digital Trends - How DoublePoint Gives Smartwatches Superpowers](https://www.digitaltrends.com/phones/wear-os-smartwatches-amazing-gesture-system-doublepoint-ces-2024/)
- TouchSDK (open source) - Reference implementation for IMU-based gestures

### 7.4 Additional Reading

**Quaternion Math:**
- [Understanding Quaternions in IMU Technology](https://daischsensor.com/understanding-quaternions-in-imu-technology/)
- [Quaternions for Orientation - Endaq Blog](https://blog.endaq.com/quaternions-for-orientation)

**Control Systems:**
- [Gaming Controller Sensitivity Settings Explained](https://medium.com/@muhammad_faizan2/gaming-controller-sensitivity-settings-explained-810819b80255)
- [Hall Effect Controller Calibration](https://joltfly.com/hall-effect-controller-calibration-the-jitter-fix-guide/)

---

## Summary

This research document provides comprehensive algorithms and implementation guidance for motion-to-cursor control using smartwatch IMU data:

1. **Sensor Fusion**: Madgwick filter recommended for best balance of accuracy and efficiency
2. **Positioning Modes**: Start with relative positioning for robustness, add hybrid for advanced control
3. **Drift Correction**: Primary defense is sensor fusion; add ZUPT and periodic recalibration
4. **User Experience**: One Euro filter for smoothing, power curves for sensitivity, radial dead zones
5. **Gesture Detection**: State machine approach with threshold-based detection; ML optional for production

**Quick Start Recommendation:**
- Madgwick filter (β=0.1)
- Relative positioning with dead zone (2.0°)
- One Euro filter (mincutoff=1.2)
- Power curve sensitivity (power=2.0, knee=0.5)
- FSM-based gesture recognition

This provides a solid foundation for a responsive, accurate, and fatigue-free cursor control system.
