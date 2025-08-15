from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.db import connection
from landing.models import UserProfile, SystemSettings
from django.core.management import call_command

class Command(BaseCommand):
    help = 'Set up default database settings and users'

    def handle(self, *args, **options):
        self.stdout.write('🚀 Setting up default database configuration...')
        
        # Create database tables if they don't exist
        try:
            self.stdout.write('📊 Creating database tables...')
            call_command('migrate', verbosity=0)
            self.stdout.write(self.style.SUCCESS('✅ Database tables created successfully'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error creating tables: {e}'))
            return
        
        # Create default superuser if none exists
        if not User.objects.filter(is_superuser=True).exists():
            self.stdout.write('👑 Creating default superuser...')
            try:
                # Create superuser with is_superuser=True
                superuser = User.objects.create_user(
                    username='admin',
                    email='admin@example.com',
                    password='Admin123!',
                    is_superuser=True,
                    is_staff=True,
                    is_active=True
                )
                
                # Create profile for superuser
                profile, created = UserProfile.objects.get_or_create(user=superuser)
                profile.user_level = 'admin'
                profile.is_verified = True
                profile.save()
                
                self.stdout.write(self.style.SUCCESS(f'✅ Default superuser created: admin/Admin123!'))
                self.stdout.write(self.style.SUCCESS(f'✅ is_superuser=True set for admin user'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'❌ Error creating superuser: {e}'))
        else:
            self.stdout.write('👑 Superuser already exists, skipping creation')
        
        # Create default basic user if none exists
        if not User.objects.filter(is_superuser=False).exists():
            self.stdout.write('👤 Creating default basic user...')
            try:
                # Create basic user with is_superuser=False
                basic_user = User.objects.create_user(
                    username='user',
                    email='user@example.com',
                    password='User123!',
                    is_superuser=False,
                    is_staff=False,
                    is_active=True
                )
                
                # Create profile for basic user
                profile, created = UserProfile.objects.get_or_create(user=basic_user)
                profile.user_level = 'basic'
                profile.is_verified = False
                profile.save()
                
                self.stdout.write(self.style.SUCCESS(f'✅ Default basic user created: user/User123!'))
                self.stdout.write(self.style.SUCCESS(f'✅ is_superuser=False set for basic user'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'❌ Error creating basic user: {e}'))
        else:
            self.stdout.write('👤 Basic users already exist, skipping creation')
        
        # Create default system settings
        self.stdout.write('⚙️ Creating default system settings...')
        default_settings = [
            ('system_name', 'Fraud Detection System', 'Name of the system'),
            ('default_user_level', 'basic', 'Default user level for new users'),
            ('auto_verify_users', 'false', 'Whether to auto-verify new users'),
            ('max_predictions_per_user', '1000', 'Maximum predictions allowed per user'),
        ]
        
        for key, value, description in default_settings:
            setting, created = SystemSettings.objects.get_or_create(
                key=key,
                defaults={'value': value, 'description': description}
            )
            if created:
                self.stdout.write(f'✅ Created setting: {key} = {value}')
            else:
                self.stdout.write(f'ℹ️ Setting already exists: {key}')
        
        # Display current user status
        self.stdout.write('\n📊 Current User Status:')
        superusers = User.objects.filter(is_superuser=True)
        basic_users = User.objects.filter(is_superuser=False)
        
        for user in superusers:
            profile = getattr(user, 'profile', None)
            level = profile.user_level if profile else 'No Profile'
            self.stdout.write(f'👑 Superuser: {user.username} (is_superuser=True, level={level})')
        
        for user in basic_users:
            profile = getattr(user, 'profile', None)
            level = profile.user_level if profile else 'No Profile'
            self.stdout.write(f'👤 Basic User: {user.username} (is_superuser=False, level={level})')
        
        self.stdout.write('\n🎉 Database setup completed successfully!')
        self.stdout.write('🔑 Default credentials:')
        self.stdout.write('   Admin: admin/Admin123! (is_superuser=True)')
        self.stdout.write('   User: user/User123! (is_superuser=False)')
