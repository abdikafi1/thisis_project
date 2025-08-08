from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from landing.models import UserProfile

class Command(BaseCommand):
    help = 'Create user profiles for existing users who don\'t have one'

    def handle(self, *args, **options):
        users_without_profile = []
        created_count = 0
        
        for user in User.objects.all():
            try:
                # Check if user has a profile
                user.profile
            except UserProfile.DoesNotExist:
                # Create profile for user
                UserProfile.objects.create(user=user)
                users_without_profile.append(user.username)
                created_count += 1
        
        if created_count > 0:
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully created {created_count} user profiles for: {", ".join(users_without_profile)}'
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS('All users already have profiles.')
            )
