# Security Topics Reference

## Zero Trust Architecture

### Definition
Zero Trust is a security framework that requires all users, whether inside or outside the organization's network, to be authenticated, authorized, and continuously validated before being granted access to applications and data.

### Core Principles
1. **Never trust, always verify** - No implicit trust based on network location
2. **Assume breach** - Design systems as if attackers are already inside
3. **Least privilege access** - Give users only the access they need, nothing more
4. **Micro-segmentation** - Divide networks into small zones to contain breaches

### Why Identity is Central
In Zero Trust, identity is the new perimeter. With employees working from anywhere and applications hosted everywhere, you can't rely on firewalls to define "inside" and "outside." Instead:
- Identity verifies WHO is requesting access
- Device trust verifies WHAT device they're using
- Context determines WHETHER to grant access
- Continuous monitoring ensures trust is maintained

### Industry Adoption
- 76% of organizations have begun implementing Zero Trust (2024 survey)
- NIST SP 800-207 provides the US government standard
- Zero Trust reduces breach impact by an average of 40%

## Phishing and Credential Attacks

### Current Threat Landscape
- 80% of breaches involve compromised credentials
- Phishing attacks increased 47% year-over-year
- Average cost of a credential-based breach: $4.5 million

### Why MFA Matters
Multi-factor authentication stops 99.9% of automated credential attacks. When attackers steal a password, MFA ensures they still can't get in without:
- Something you have (phone, hardware key)
- Something you are (biometric)
- Somewhere you are (location context)

### Adaptive Authentication
Not all logins are equal. Adaptive authentication adjusts requirements based on:
- Login location and device
- Time of day and behavior patterns
- Sensitivity of the requested resource
- Real-time threat intelligence

## Compliance Frameworks

### SOC 2
Service Organization Control 2 evaluates security, availability, processing integrity, confidentiality, and privacy. Key for SaaS vendors.

### ISO 27001
International standard for information security management systems (ISMS). Demonstrates systematic approach to managing sensitive information.

### HIPAA
Health Insurance Portability and Accountability Act requires healthcare organizations to protect patient data. Strong access controls are essential.

### FedRAMP
Federal Risk and Authorization Management Program standardizes security assessment for cloud services used by US government agencies.

## Identity Lifecycle

### Joiner
When an employee joins, they need access to:
- Email and collaboration tools (day 1)
- Role-specific applications (day 1-7)
- Specialized systems (as needed)

Automation reduces provisioning from 2 weeks to minutes.

### Mover
Role changes require access updates:
- New permissions for new role
- Revocation of old permissions
- Manager approval workflows

Without automation, movers often accumulate excessive access.

### Leaver
When someone leaves, all access must be revoked:
- Immediately for terminated employees
- On departure date for voluntary exits
- Including third-party SaaS apps often forgotten

Average time to deprovision without automation: 3+ days. With automation: instant.
