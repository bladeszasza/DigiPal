# DigiPal User Guide

## Welcome to DigiPal! 🎮

DigiPal is your digital companion that combines the nostalgic charm of classic virtual pets with cutting-edge AI technology. This guide will help you get started with raising, caring for, and bonding with your DigiPal.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Your First DigiPal](#your-first-digipal)
3. [Understanding Your Pet](#understanding-your-pet)
4. [Care and Training](#care-and-training)
5. [Communication](#communication)
6. [Evolution System](#evolution-system)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for AI models and authentication

### Installation

#### Option 1: Docker (Recommended)

1. **Install Docker**: Download from [docker.com](https://docker.com)

2. **Run DigiPal**:
   ```bash
   docker run -p 7860:7860 digipal/digipal:latest
   ```

3. **Access the Interface**: Open http://localhost:7860 in your browser

#### Option 2: Local Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/digipal.git
   cd digipal
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Launch DigiPal**:
   ```bash
   python launch_digipal.py
   ```

### First-Time Setup

1. **HuggingFace Authentication**: You'll need a free HuggingFace account
   - Visit [huggingface.co](https://huggingface.co) to create an account
   - Generate an access token in your settings
   - Enter the token when prompted in DigiPal

2. **Choose Your Environment**:
   - **Online Mode**: Full AI features with cloud models
   - **Offline Mode**: Limited features for development/testing

## Your First DigiPal

### Authentication

1. Open DigiPal in your browser
2. Navigate to the **Authentication** tab
3. Enter your HuggingFace token
4. Click **Login**

### Egg Selection

After successful authentication, you'll choose your DigiPal's egg:

#### 🔴 Red Egg (Fire Type)
- **Specialization**: Offense and attack power
- **Personality**: Energetic and fierce
- **Evolution Bonus**: +25% offense attributes
- **Best For**: Players who enjoy training and battles

#### 🔵 Blue Egg (Water Type)
- **Specialization**: Defense and magic power
- **Personality**: Calm and protective
- **Evolution Bonus**: +25% defense and MP attributes
- **Best For**: Players who prefer strategic gameplay

#### 🟢 Green Egg (Earth Type)
- **Specialization**: Health and symbiosis
- **Personality**: Wise and nurturing
- **Evolution Bonus**: +25% HP and endurance attributes
- **Best For**: Players who enjoy long-term care and bonding

### Hatching Your DigiPal

1. After selecting an egg, proceed to the **Main Interface**
2. Use the **Voice Input** or **Text Chat** to speak to your egg
3. Say anything - even "Hello!" will trigger hatching
4. Watch as your DigiPal emerges as a baby!

## Understanding Your Pet

### Life Stages

Your DigiPal will progress through seven life stages:

1. **Egg** (30 minutes): Waiting to hatch
2. **Baby** (1 day): Basic needs, simple commands
3. **Child** (3 days): Learning and growth
4. **Teen** (5 days): Developing personality
5. **Young Adult** (7 days): Peak learning ability
6. **Adult** (10 days): Full capabilities
7. **Elderly** (3 days): Wisdom and preparation for next generation

### Attributes

Your DigiPal has several key attributes that affect its behavior and abilities:

#### Primary Attributes
- **HP (Health Points)**: Overall health and vitality
- **MP (Magic Points)**: Energy for special abilities
- **Offense**: Attack power and strength
- **Defense**: Protection and resilience
- **Speed**: Agility and reaction time
- **Brains**: Intelligence and learning ability

#### Secondary Attributes
- **Discipline**: Training effectiveness and obedience
- **Happiness**: Mood and willingness to interact
- **Weight**: Physical condition affecting performance
- **Energy**: Current stamina level
- **Care Mistakes**: Accumulated poor care decisions

### Status Indicators

The interface shows real-time status information:

- **Needs Attention**: 🚨 Your pet requires immediate care
- **Evolution Ready**: ⭐ Ready to evolve to the next stage
- **Happy**: 😊 High happiness level
- **Tired**: 😴 Low energy, needs rest
- **Hungry**: 🍽️ Needs feeding

## Care and Training

### Basic Care Actions

#### 🍖 Feeding
Keep your DigiPal nourished with various food types:

- **Meat**: +2 Weight, +1 HP, +1 Offense, +5 Happiness
- **Fish**: +1 Weight, +1 Brains, +1 MP, +3 Happiness
- **Vegetables**: +1 Weight, +1 Defense, +2 Happiness
- **Protein Shake**: +1 Weight, +2 Offense, +1 HP, +1 Happiness
- **Energy Drink**: +5 Energy, +1 Speed, +1 MP, +3 Happiness

#### 💪 Training
Develop your DigiPal's abilities through specialized training:

**Basic Training Options:**
- **Strength Training**: +3 Offense, +2 HP, -1 Weight
- **Defense Training**: +3 Defense, +2 HP, -1 Weight
- **Speed Training**: +3 Speed, -2 Weight
- **Brain Training**: +3 Brains, +2 MP
- **Endurance Training**: +4 HP, +2 Defense, -2 Weight
- **Agility Training**: +4 Speed, +1 Offense, -3 Weight

#### 💝 Social Care
Maintain your DigiPal's emotional well-being:

- **Praise** (👍): +10 Happiness, -2 Discipline
- **Scold** (👎): -8 Happiness, +5 Discipline
- **Play** (🎾): +8 Happiness, -1 Weight, -8 Energy
- **Rest** (😴): +30 Energy, +3 Happiness

#### 🧼 Maintenance
Keep your DigiPal healthy and clean:

- **Clean**: +5 Happiness, removes negative status effects
- **Medicine**: Restores health, cures illnesses

### Training Strategy

#### For Beginners
1. **Focus on Happiness**: Keep happiness above 50 for better training results
2. **Balanced Training**: Don't neglect any single attribute
3. **Regular Feeding**: Maintain energy levels for effective training
4. **Rest When Needed**: Tired pets train poorly

#### For Advanced Users
1. **Specialization**: Focus on your egg type's strengths
2. **Evolution Planning**: Train specific attributes for desired evolution paths
3. **Weight Management**: Monitor weight for optimal performance
4. **Discipline Balance**: High discipline improves training but may reduce happiness

## Communication

### Text Chat

Use the text input to communicate with your DigiPal:

1. Type your message in the text box
2. Click **Send** or press Enter
3. Your DigiPal will respond based on its current stage and personality

### Voice Interaction

For a more immersive experience, use voice commands:

1. Click the **🎤 Record** button
2. Speak clearly into your microphone
3. Click **Process Speech** to convert to text
4. Your DigiPal will respond to the interpreted command

### Command Understanding

Your DigiPal's command understanding evolves with its life stage:

#### Baby Stage
- Basic commands: "eat", "sleep", "good", "bad"
- Simple emotional responses
- Limited vocabulary

#### Child Stage
- Expanded commands: "train", "play", "clean"
- Beginning to understand complex sentences
- Shows personality traits

#### Teen and Beyond
- Full command understanding
- Complex conversation ability
- Personality-based responses
- Memory of past interactions

### Quick Messages

Use the quick message buttons for common interactions:
- **👋 Hello**: Friendly greeting
- **❓ How are you?**: Check pet's mood and status

## Evolution System

### Evolution Requirements

Each life stage has specific requirements:

#### Baby → Child
- **Time**: 24 hours minimum
- **Happiness**: Above 30
- **Care Mistakes**: Less than 5

#### Child → Teen
- **Time**: 3 days minimum
- **Attributes**: At least 20 in primary stats
- **Training**: Minimum 10 training sessions

#### Teen → Young Adult
- **Time**: 5 days minimum
- **Attributes**: At least 40 in primary stats
- **Happiness**: Above 50
- **Discipline**: Above 30

#### Young Adult → Adult
- **Time**: 7 days minimum
- **Attributes**: At least 60 in primary stats
- **Specialization**: High stats in egg type specialty

#### Adult → Elderly
- **Time**: 10 days minimum
- **Mastery**: All attributes above 50
- **Care Quality**: Excellent care rating

### Evolution Bonuses

When your DigiPal evolves, it receives significant bonuses:

- **Attribute Increases**: Based on egg type and care quality
- **New Abilities**: Expanded command understanding
- **Visual Changes**: Updated appearance reflecting growth
- **Personality Development**: More complex interactions

### Manual Evolution

You can trigger evolution manually if requirements are met:
1. Check the **Evolution Ready** indicator
2. Use the MCP API or wait for automatic evolution
3. Celebrate your DigiPal's growth!

## Advanced Features

### Generational Inheritance

When your DigiPal reaches the end of its life cycle:

1. **DNA Creation**: Final attributes determine inheritance bonuses
2. **New Egg**: Receive a new egg with inherited traits
3. **Attribute Bonuses**: Based on parent's care quality:
   - Perfect Care: +25% attribute bonus
   - Excellent Care: +20% attribute bonus
   - Good Care: +15% attribute bonus
   - Fair Care: +10% attribute bonus
   - Poor Care: +5% attribute bonus

### Memory System

Your DigiPal remembers:
- **Conversation History**: Past interactions and topics
- **Care Patterns**: Your preferred care methods
- **Emotional Memories**: Positive and negative experiences
- **Learning Progress**: Commands and behaviors learned

### Image Generation

Your DigiPal's appearance is dynamically generated:
- **Life Stage Reflection**: Visual changes with evolution
- **Attribute Influence**: Stats affect appearance
- **Personality Traits**: Happiness and care quality visible
- **Egg Type Themes**: Elemental characteristics maintained

## Troubleshooting

### Common Issues

#### DigiPal Won't Respond
1. Check internet connection
2. Verify HuggingFace token is valid
3. Ensure microphone permissions (for voice)
4. Try refreshing the browser

#### Evolution Not Occurring
1. Check all evolution requirements are met
2. Verify sufficient time has passed
3. Ensure happiness and care levels are adequate
4. Try manual evolution trigger

#### Poor Performance
1. Close other browser tabs
2. Check available system memory
3. Restart the application
4. Consider using offline mode

#### Authentication Issues
1. Verify HuggingFace token is correct
2. Check token permissions and expiration
3. Try logging out and back in
4. Use offline mode for testing

### Getting Help

1. **Check Logs**: Look in the logs directory for error messages
2. **Health Check**: Visit `/health` endpoint for system status
3. **Documentation**: Refer to technical documentation
4. **Community**: Join the DigiPal community for support

### Performance Tips

1. **Regular Interaction**: Daily interaction keeps your DigiPal happy
2. **Balanced Care**: Don't focus on just one attribute
3. **Monitor Status**: Watch for attention indicators
4. **Plan Evolution**: Prepare for evolution requirements in advance
5. **Backup Data**: Your progress is automatically saved

## Conclusion

DigiPal offers a unique blend of nostalgic virtual pet mechanics and modern AI technology. Take your time to explore, experiment, and most importantly, enjoy bonding with your digital companion!

Remember: Every DigiPal is unique, shaped by your care and interaction style. There's no single "correct" way to raise your pet - find what works for you and your DigiPal's personality.

Happy pet raising! 🎮✨